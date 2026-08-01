use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serenity::all::{
    ChannelId, Command, CommandInteraction, CommandOptionType, CreateCommand,
    CreateCommandOption, CreateInteractionResponse, CreateInteractionResponseMessage,
    EditInteractionResponse, GuildId, Message, UserId,
};
use serenity::prelude::Context;
use tokio::sync::RwLock;

use crate::config::AppConfig;
use crate::gemini::ask_gemini;
use crate::transcription::{
    AsrEngine, SessionSenders, SsrcMap, Streams,
};

mod commands;
mod session;

pub(super) const LOG_DEFAULT_UTTERANCES: i64 = 40;
pub(super) const LOG_MAX_DISCORD_CHARS: usize = 1800;
pub(super) const THREAD_CONTEXT_MAX_ENTRIES: usize = 64;
pub(super) const THREAD_HISTORY_MAX_ITEMS: usize = 24;
pub(super) const FINALIZE_SETTLE_TIMEOUT: Duration = Duration::from_millis(900);
pub(super) const FINALIZE_SETTLE_PASSES: usize = 4;
pub(super) const STARTUP_RECEIVE_WATCHDOG_DELAY: Duration = Duration::from_secs(10);
pub(super) const STARTUP_RECEIVE_RECOVERY_MAX_ATTEMPTS: u8 = 3;
pub(super) const STEADY_STATE_WATCHDOG_CADENCE: Duration = Duration::from_secs(30);
pub(super) const STEADY_STATE_NO_PROGRESS_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone)]
pub struct Utterance {
    pub user_id: UserId,
    pub start_ts: Instant,
    pub text: String,
}

pub struct CallSession {
    pub voice_channel: ChannelId,
    pub text_channel: ChannelId,
    pub transcript: Vec<Utterance>,
    pub transcript_jsonl_path: std::path::PathBuf,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub started_mono: Instant,
}

pub struct ThreadContext {
    pub transcript: String,
    pub history: Vec<(String, String)>,
}

pub struct AppState {
    pub active_calls: DashMap<GuildId, Arc<RwLock<CallSession>>>,
    pub transcript_threads: DashMap<ChannelId, ThreadContext>,
    pub thread_context_last_used: DashMap<ChannelId, Instant>,
    pub gemini_key: String,
    pub gemini_model: String,
    pub live_transcript_debug: bool,
    pub enable_denoiser: bool,
    pub endpoint_silence_ticks: u32,
    pub rolling_ingest_max_ms: u64,
    pub rolling_ingest_context_ms: u64,
    pub autojoin_suffix: String,
    pub asr: Arc<AsrEngine>,
    pub ssrc_to_user: Arc<SsrcMap>,
    pub streams: Arc<Streams>,
    pub recovery_locks: DashMap<GuildId, Arc<tokio::sync::Mutex<()>>>,
    pub utterance_senders: SessionSenders,
    pub transcription_inflight: DashMap<GuildId, Arc<AtomicUsize>>,
    pub transcript_pending_commits: DashMap<GuildId, Arc<AtomicUsize>>,
    pub decode_shed_total: DashMap<GuildId, Arc<AtomicUsize>>,
    pub resample_error_total: DashMap<GuildId, Arc<AtomicUsize>>,
    pub decoded_audio_activity: DashMap<GuildId, Arc<AtomicUsize>>,
    pub decode_failure_activity: DashMap<GuildId, Arc<AtomicUsize>>,
    pub unmapped_ssrc_activity: DashMap<GuildId, Arc<AtomicUsize>>,
    pub transcription_started_notified: DashMap<GuildId, Arc<AtomicBool>>,
}

impl AppState {
    pub fn new(cfg: AppConfig) -> anyhow::Result<Self> {
        let asr = Arc::new(AsrEngine::new(&cfg.asr_model_dir, cfg.asr_num_threads)?);

        Ok(Self {
            active_calls: DashMap::new(),
            transcript_threads: DashMap::new(),
            thread_context_last_used: DashMap::new(),
            gemini_key: cfg.gemini_api_key,
            gemini_model: cfg.gemini_model,
            live_transcript_debug: cfg.live_transcript_debug,
            enable_denoiser: cfg.enable_denoiser,
            endpoint_silence_ticks: ((cfg.endpoint_silence_ms.saturating_add(19) / 20) as u32).max(1),
            rolling_ingest_max_ms: cfg.rolling_ingest_max_ms,
            rolling_ingest_context_ms: cfg.rolling_ingest_context_ms,
            autojoin_suffix: cfg.autojoin_suffix,
            asr,
            ssrc_to_user: Arc::new(DashMap::new()),
            streams: Arc::new(DashMap::new()),
            recovery_locks: DashMap::new(),
            utterance_senders: DashMap::new(),
            transcription_inflight: DashMap::new(),
            transcript_pending_commits: DashMap::new(),
            decode_shed_total: DashMap::new(),
            resample_error_total: DashMap::new(),
            decoded_audio_activity: DashMap::new(),
            decode_failure_activity: DashMap::new(),
            unmapped_ssrc_activity: DashMap::new(),
            transcription_started_notified: DashMap::new(),
        })
    }
}

fn touch_thread_context(state: &Arc<AppState>, channel_id: ChannelId) {
    state
        .thread_context_last_used
        .insert(channel_id, Instant::now());
}

fn evict_thread_contexts_if_needed(state: &Arc<AppState>) {
    while state.transcript_threads.len() > THREAD_CONTEXT_MAX_ENTRIES {
        let oldest = state
            .thread_context_last_used
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .min_by_key(|(_, ts)| *ts)
            .map(|(channel_id, _)| channel_id);

        let Some(channel_id) = oldest else {
            break;
        };

        state.transcript_threads.remove(&channel_id);
        state.thread_context_last_used.remove(&channel_id);
    }
}

pub(super) fn upsert_thread_context(
    state: &Arc<AppState>,
    channel_id: ChannelId,
    transcript: String,
) {
    state.transcript_threads.insert(
        channel_id,
        ThreadContext {
            transcript,
            history: Vec::new(),
        },
    );
    touch_thread_context(state, channel_id);
    evict_thread_contexts_if_needed(state);
}

pub async fn register_commands(ctx: &Context) -> anyhow::Result<()> {
    let cmds = vec![
        CreateCommand::new("join").description("Join your current voice channel in this guild"),
        CreateCommand::new("leave").description("Leave voice and finalize transcript export"),
        CreateCommand::new("status").description("Show live transcription status for this guild"),
        CreateCommand::new("log")
            .description("Show recent committed transcript lines for the active call")
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::Integer,
                    "utterances",
                    "How many recent utterances to include (default: 40)",
                )
                .required(false),
            ),
        CreateCommand::new("ask")
            .description("Ask about the current call transcript")
            .add_option(
                CreateCommandOption::new(CommandOptionType::String, "question", "Question to ask")
                    .required(true),
            ),
        CreateCommand::new("autojoin")
            .description("Mark your current voice channel for automatic future joins")
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::Channel,
                    "channel",
                    "Voice channel to mark (defaults to your current voice channel)",
                )
                .required(false),
            )
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::Boolean,
                    "enabled",
                    "Set true to enable, false to disable (omit to toggle)",
                )
                .required(false),
            ),
    ];

    Command::set_global_commands(&ctx.http, cmds).await?;
    Ok(())
}

pub async fn handle_slash_command(
    ctx: &Context,
    state: &Arc<AppState>,
    command: CommandInteraction,
) {
    let _ = command
        .create_response(
            &ctx.http,
            CreateInteractionResponse::Defer(CreateInteractionResponseMessage::new()),
        )
        .await;

    let result = match command.data.name.as_str() {
        "join" => commands::handle_join(ctx, state, &command).await,
        "leave" => commands::handle_leave(ctx, state, &command).await,
        "status" => commands::handle_status(ctx, state, &command).await,
        "log" => commands::handle_log(ctx, state, &command).await,
        "ask" => commands::handle_ask(ctx, state, &command).await,
        "autojoin" => commands::handle_autojoin(ctx, state, &command).await,
        _ => Ok("Unknown command".to_string()),
    };

    let content = match result {
        Ok(msg) => msg,
        Err(err) => format!("error: {err:#}"),
    };

    let _ = command
        .edit_response(&ctx.http, EditInteractionResponse::new().content(content))
        .await;
}

pub async fn handle_message(ctx: &Context, state: &Arc<AppState>, msg: Message) {
    if msg.author.bot {
        return;
    }

    if !state.transcript_threads.contains_key(&msg.channel_id) {
        if let Err(e) = session::maybe_load_thread_context(ctx, state, msg.channel_id).await {
            tracing::warn!("failed to lazy-load thread transcript context: {e:#}");
        }
    }

    if let Some(thread_ctx) = state.transcript_threads.get(&msg.channel_id) {
        touch_thread_context(state, msg.channel_id);
        let question = msg.content.clone();
        let transcript = thread_ctx.transcript.clone();
        let prior_turns = thread_ctx.history.clone();
        drop(thread_ctx);

        let answer = ask_gemini(
            &state.gemini_key,
            &state.gemini_model,
            &transcript,
            &question,
            Some(&prior_turns),
        )
        .await
        .unwrap_or_else(|e| format!("gemini error: {e}"));

        let _ = msg.channel_id.say(&ctx.http, &answer).await;

        if let Some(mut thread_ctx) = state.transcript_threads.get_mut(&msg.channel_id) {
            thread_ctx.history.push(("user".to_string(), question));
            thread_ctx.history.push(("model".to_string(), answer));
            if thread_ctx.history.len() > THREAD_HISTORY_MAX_ITEMS {
                let remove = thread_ctx.history.len() - THREAD_HISTORY_MAX_ITEMS;
                thread_ctx.history.drain(0..remove);
            }
        }

        touch_thread_context(state, msg.channel_id);
        evict_thread_contexts_if_needed(state);
    }
}

pub use session::maybe_finalize_on_empty_voice_channel;
pub use commands::maybe_autojoin_on_voice_state;
