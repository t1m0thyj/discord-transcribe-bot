use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serenity::all::{
    AutocompleteChoice, ChannelId, Command, CommandDataOptionValue, CommandInteraction,
    CommandOptionType, CreateAllowedMentions, CreateAutocompleteResponse, CreateCommand,
    CreateCommandOption, CreateInteractionResponse, CreateInteractionResponseMessage,
    EditInteractionResponse, GuildId, Message, Permissions, UserId,
};
use serenity::prelude::Context;
use tokio::sync::mpsc;
use tokio::sync::RwLock;

use crate::config::AppConfig;
use crate::ai::AiClient;
use crate::asr::{
    AsrEngine, SsrcMap, Streams,
};

mod autojoin;
mod commands;
mod journal;
mod session;
mod summary;

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
pub(super) const THREAD_AI_COOLDOWN: Duration = Duration::from_secs(3);
const ASK_PROMPT_HISTORY_MAX_ITEMS: usize = 10;
const ASK_AUTOCOMPLETE_MAX_CHARS: usize = 100;

fn endpoint_silence_ticks(endpoint_silence_ms: u64) -> u32 {
    ((endpoint_silence_ms.saturating_add(19) / 20) as u32).max(1)
}

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

pub struct GuildRuntime {
    pub utterance_tx: mpsc::Sender<Utterance>,
    pub transcription_inflight: AtomicUsize,
    pub transcript_pending_commits: AtomicUsize,
    pub decode_jobs_total: AtomicUsize,
    pub decode_jobs_with_text: AtomicUsize,
    pub asr_decode_error_total: AtomicUsize,
    pub decode_audio_total_ms: AtomicUsize,
    pub decode_total_ms: AtomicUsize,
    pub decode_queue_wait_total_ms: AtomicUsize,
    pub decode_last_ms: AtomicUsize,
    pub decode_queue_wait_last_ms: AtomicUsize,
    pub decode_shed_total: AtomicUsize,
    pub dispatch_gate_total: AtomicUsize,
    pub resample_error_total: AtomicUsize,
    pub decoded_audio_activity: AtomicUsize,
    pub decode_failure_activity: AtomicUsize,
    pub unmapped_ssrc_activity: AtomicUsize,
    pub transcription_started_notified: AtomicBool,
    pub recovery_lock: Arc<tokio::sync::Mutex<()>>,
}

impl GuildRuntime {
    pub fn new(utterance_tx: mpsc::Sender<Utterance>) -> Self {
        Self {
            utterance_tx,
            transcription_inflight: AtomicUsize::new(0),
            transcript_pending_commits: AtomicUsize::new(0),
            decode_jobs_total: AtomicUsize::new(0),
            decode_jobs_with_text: AtomicUsize::new(0),
            asr_decode_error_total: AtomicUsize::new(0),
            decode_audio_total_ms: AtomicUsize::new(0),
            decode_total_ms: AtomicUsize::new(0),
            decode_queue_wait_total_ms: AtomicUsize::new(0),
            decode_last_ms: AtomicUsize::new(0),
            decode_queue_wait_last_ms: AtomicUsize::new(0),
            decode_shed_total: AtomicUsize::new(0),
            dispatch_gate_total: AtomicUsize::new(0),
            resample_error_total: AtomicUsize::new(0),
            decoded_audio_activity: AtomicUsize::new(0),
            decode_failure_activity: AtomicUsize::new(0),
            unmapped_ssrc_activity: AtomicUsize::new(0),
            transcription_started_notified: AtomicBool::new(false),
            recovery_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }
}

pub struct AppState {
    pub active_calls: DashMap<GuildId, Arc<RwLock<CallSession>>>,
    pub session_start_locks: DashMap<GuildId, Arc<tokio::sync::Mutex<()>>>,
    pub transcript_threads: DashMap<ChannelId, ThreadContext>,
    pub thread_context_last_used: DashMap<ChannelId, Instant>,
    pub thread_ai_last_reply: DashMap<ChannelId, Instant>,
    pub ask_prompt_history: DashMap<(UserId, Option<GuildId>), Vec<String>>,
    pub ai: Arc<AiClient>,
    pub live_transcript_debug: bool,
    pub enable_denoiser: bool,
    pub endpoint_silence_ticks: u32,
    pub rolling_ingest_max_ms: u64,
    pub rolling_ingest_context_ms: u64,
    pub transcript_retention_days: u64,
    pub autojoin_suffix: String,
    pub post_call_summary_enabled: bool,
    pub post_call_summary_post_in_thread: bool,
    pub post_call_summary_include_in_markdown: bool,
    pub asr: Arc<AsrEngine>,
    pub ssrc_to_user: Arc<SsrcMap>,
    pub streams: Arc<Streams>,
    pub guild_runtimes: DashMap<GuildId, Arc<GuildRuntime>>,
}

impl AppState {
    pub fn new(cfg: AppConfig) -> anyhow::Result<Self> {
        let asr = Arc::new(AsrEngine::new(
            &cfg.asr.model_dir,
            cfg.asr.num_threads,
            cfg.asr.model_family.as_deref(),
        )?);
        let ai = Arc::new(AiClient::new(
            cfg.ai.provider.clone(),
            cfg.ai.request_timeout,
        )?);
        tracing::info!(
            provider = ai.provider_label(),
            request_timeout = cfg.ai.request_timeout,
            "AI provider configured"
        );

        Ok(Self {
            active_calls: DashMap::new(),
            session_start_locks: DashMap::new(),
            transcript_threads: DashMap::new(),
            thread_context_last_used: DashMap::new(),
            thread_ai_last_reply: DashMap::new(),
            ask_prompt_history: DashMap::new(),
            ai,
            live_transcript_debug: cfg.debug.log_live_transcript,
            enable_denoiser: cfg.audio.enable_denoiser,
            // Endpoint uses VAD hangover plus silence ticks; effective trailing wait
            // is roughly (endpoint_silence_ticks * 20ms) + ~256ms.
            endpoint_silence_ticks: endpoint_silence_ticks(cfg.transcription.endpoint_silence_ms),
            rolling_ingest_max_ms: cfg.transcription.rolling_ingest_max_ms,
            rolling_ingest_context_ms: cfg.transcription.rolling_ingest_context_ms,
            transcript_retention_days: cfg.transcription.retention_days,
            autojoin_suffix: cfg.discord.autojoin_suffix,
            post_call_summary_enabled: cfg.summary.enabled,
            post_call_summary_post_in_thread: cfg.summary.post_in_thread,
            post_call_summary_include_in_markdown: cfg.summary.include_in_markdown,
            asr,
            ssrc_to_user: Arc::new(DashMap::new()),
            streams: Arc::new(DashMap::new()),
            guild_runtimes: DashMap::new(),
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
        state.thread_ai_last_reply.remove(&channel_id);
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
        CreateCommand::new("join")
            .description("Join your current voice channel in this guild")
            .default_member_permissions(Permissions::MOVE_MEMBERS),
        CreateCommand::new("leave")
            .description("Leave voice and finalize transcript export")
            .default_member_permissions(Permissions::MOVE_MEMBERS),
        CreateCommand::new("status")
            .description("Show live transcription status for this guild"),
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
                    .set_autocomplete(true)
                .required(true),
            ),
        CreateCommand::new("summary")
            .description("Summarize the current call transcript"),
        CreateCommand::new("autojoin")
            .description("Mark your current voice channel for automatic future joins")
            .default_member_permissions(Permissions::MANAGE_CHANNELS)
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
    let command_text = format_slash_command(&command);
    remember_ask_prompt(state, &command);
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
        "summary" => commands::handle_summary(ctx, state, &command).await,
        "autojoin" => commands::handle_autojoin(ctx, state, &command).await,
        _ => Ok("Unknown command".to_string()),
    };

    let result = match result {
        Ok(msg) => msg,
        Err(err) => format!("error: {err:#}"),
    };
    let content = format!("**Command:** {command_text}\n\n{result}");

    let _ = command
        .edit_response(
            &ctx.http,
            EditInteractionResponse::new()
                .content(content)
                .allowed_mentions(CreateAllowedMentions::new()),
        )
        .await;
}

pub async fn handle_autocomplete(
    ctx: &Context,
    state: &Arc<AppState>,
    command: CommandInteraction,
) {
    let input = command
        .data
        .options
        .iter()
        .find_map(|option| match &option.value {
            CommandDataOptionValue::Autocomplete { value, .. } if option.name == "question" => {
                Some(value.as_str())
            }
            _ => None,
        })
        .unwrap_or_default();

    let choices = if command.data.name == "ask" {
        state
            .ask_prompt_history
            .get(&(command.user.id, command.guild_id))
            .map(|history| autocomplete_prompt_choices(&history, input))
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    if let Err(error) = command
        .create_response(
            &ctx.http,
            CreateInteractionResponse::Autocomplete(
                CreateAutocompleteResponse::new().set_choices(choices),
            ),
        )
        .await
    {
        tracing::warn!(error = %error, "failed to send command autocomplete choices");
    }
}

fn remember_ask_prompt(state: &Arc<AppState>, command: &CommandInteraction) {
    if command.data.name != "ask" {
        return;
    }

    let Some(prompt) = command
        .data
        .options
        .iter()
        .find(|option| option.name == "question")
        .and_then(|option| match &option.value {
            CommandDataOptionValue::String(value) => Some(value.trim()),
            _ => None,
        })
        .filter(|prompt| !prompt.is_empty())
    else {
        return;
    };

    let mut history = state
        .ask_prompt_history
        .entry((command.user.id, command.guild_id))
        .or_default();
    update_prompt_history(&mut history, prompt.to_string());
}

fn update_prompt_history(history: &mut Vec<String>, prompt: String) {
    history.retain(|previous| previous != &prompt);
    history.insert(0, prompt);
    history.truncate(ASK_PROMPT_HISTORY_MAX_ITEMS);
}

fn autocomplete_prompt_choices(history: &[String], input: &str) -> Vec<AutocompleteChoice> {
    matching_prompt_values(history, input)
        .into_iter()
        .map(|prompt| AutocompleteChoice::from(prompt))
        .collect()
}

fn matching_prompt_values(history: &[String], input: &str) -> Vec<String> {
    let input = input.trim().to_lowercase();
    history
        .iter()
        .filter(|prompt| prompt.chars().count() <= ASK_AUTOCOMPLETE_MAX_CHARS)
        .filter(|prompt| input.is_empty() || prompt.to_lowercase().contains(&input))
        .take(ASK_PROMPT_HISTORY_MAX_ITEMS)
        .cloned()
        .collect()
}

fn format_slash_command(command: &CommandInteraction) -> String {
    let mut text = format!("/{}", command.data.name);
    for option in &command.data.options {
        let value = match &option.value {
            CommandDataOptionValue::String(value)
            | CommandDataOptionValue::Autocomplete { value, .. } => value.clone(),
            CommandDataOptionValue::Boolean(value) => value.to_string(),
            CommandDataOptionValue::Integer(value) => value.to_string(),
            CommandDataOptionValue::Number(value) => value.to_string(),
            CommandDataOptionValue::Channel(value) => format!("<#{}>", value.get()),
            CommandDataOptionValue::User(value) => format!("<@{}>", value.get()),
            CommandDataOptionValue::Role(value) => format!("<@&{}>", value.get()),
            CommandDataOptionValue::Mentionable(value) => value.get().to_string(),
            CommandDataOptionValue::Attachment(value) => value.get().to_string(),
            CommandDataOptionValue::SubCommand(_)
            | CommandDataOptionValue::SubCommandGroup(_)
            | CommandDataOptionValue::Unknown(_) => continue,
        };
        text.push(' ');
        text.push_str(&value);
    }
    text
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

        let now = Instant::now();
        if let Some(last) = state.thread_ai_last_reply.get(&msg.channel_id) {
            if now.saturating_duration_since(*last.value()) < THREAD_AI_COOLDOWN {
                return;
            }
        }
        state.thread_ai_last_reply.insert(msg.channel_id, now);

        let question = msg.content.clone();
        let transcript = thread_ctx.transcript.clone();
        let prior_turns = thread_ctx.history.clone();
        drop(thread_ctx);

        let typing = msg.channel_id.start_typing(&ctx.http);
        let answer = state
            .ai
            .ask(&transcript, &question, Some(&prior_turns))
        .await
        .unwrap_or_else(|e| format!("ai error: {e}"));
        drop(typing);

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

pub use autojoin::maybe_autojoin_on_voice_state;
pub use session::maybe_finalize_on_empty_voice_channel;

#[cfg(test)]
mod tests {
    use super::{
        endpoint_silence_ticks, matching_prompt_values, update_prompt_history,
        ASK_PROMPT_HISTORY_MAX_ITEMS,
    };

    #[test]
    fn endpoint_silence_ticks_rounds_up_and_never_zero() {
        assert_eq!(endpoint_silence_ticks(400), 20);
        assert_eq!(endpoint_silence_ticks(401), 21);
        assert_eq!(endpoint_silence_ticks(80), 4);
        assert_eq!(endpoint_silence_ticks(1), 1);
    }

    #[test]
    fn ask_prompt_history_deduplicates_and_keeps_most_recent_items() {
        let mut history = vec!["older".to_string(), "same".to_string()];
        update_prompt_history(&mut history, "same".to_string());
        assert_eq!(history, ["same", "older"]);

        for index in 0..ASK_PROMPT_HISTORY_MAX_ITEMS {
            update_prompt_history(&mut history, format!("prompt {index}"));
        }
        assert_eq!(history.len(), ASK_PROMPT_HISTORY_MAX_ITEMS);
        assert_eq!(
            history[0],
            format!("prompt {}", ASK_PROMPT_HISTORY_MAX_ITEMS - 1)
        );
    }

    #[test]
    fn autocomplete_matches_recent_prompts_without_returning_long_values() {
        let long_prompt = "x".repeat(101);
        let history = vec![
            "When does the project ship?".to_string(),
            "What did we decide?".to_string(),
            long_prompt,
        ];

        assert_eq!(
            matching_prompt_values(&history, "decide"),
            ["What did we decide?"]
        );
        assert_eq!(
            matching_prompt_values(&history, ""),
            ["When does the project ship?", "What did we decide?"]
        );
    }
}
