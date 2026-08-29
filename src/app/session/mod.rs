use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context as _;
use chrono::Utc;
use serenity::all::{Channel, ChannelId, ChannelType, GuildId, MessageId};
use serenity::prelude::Context;
use tokio::fs;
use tokio::sync::{mpsc, RwLock};

use super::journal::transcript_writer_loop;
use super::{AppState, CallSession, GuildRuntime, Utterance};

mod finalize;
mod watchdog;

pub use finalize::{finalize_call_for_guild, maybe_finalize_on_empty_voice_channel};
pub use watchdog::{
    attach_voice_handlers, startup_receive_watchdog, steady_state_receive_watchdog,
    VoiceHandlerAttachContext,
};

const TRANSCRIPT_ATTACHMENT_MAX_BYTES: u64 = 10 * 1024 * 1024;

pub(super) async fn start_call_session(
    ctx: &Context,
    state: &Arc<AppState>,
    guild_id: GuildId,
    voice_channel: ChannelId,
    text_channel: ChannelId,
) -> anyhow::Result<()> {
    let session_start_lock = state
        .session_start_locks
        .entry(guild_id)
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone();
    let _session_start_guard = session_start_lock.lock().await;

    if state.active_calls.contains_key(&guild_id) {
        anyhow::bail!("an active call session already exists for this guild");
    }

    let manager = songbird::get(ctx)
        .await
        .context("songbird voice manager unavailable")?
        .clone();

    let call_lock = manager
        .join(guild_id, voice_channel)
        .await
        .context("failed to join voice channel")?;

    let (utterance_tx, utterance_rx) = mpsc::channel::<Utterance>(1024);

    let session_started_at = Utc::now();
    let local_dir = PathBuf::from("transcripts");
    fs::create_dir_all(&local_dir)
        .await
        .context("failed to create local transcript directory")?;
    let transcript_jsonl_path = local_dir.join(format!(
        "transcript-{}-{}.jsonl",
        guild_id.get(),
        session_started_at.format("%Y%m%d-%H%M%S")
    ));
    fs::File::create(&transcript_jsonl_path)
        .await
        .with_context(|| {
            format!(
                "failed to create transcript journal file {}",
                transcript_jsonl_path.display()
            )
        })?;

    let runtime = Arc::new(GuildRuntime::new(utterance_tx.clone()));
    state.guild_runtimes.insert(guild_id, Arc::clone(&runtime));

    let session = Arc::new(RwLock::new(CallSession {
        voice_channel,
        text_channel,
        transcript: Vec::new(),
        transcript_jsonl_path: transcript_jsonl_path.clone(),
        started_at: session_started_at,
        started_mono: Instant::now(),
    }));
    state.active_calls.insert(guild_id, session.clone());

    let writer_runtime = Arc::clone(&runtime);
    tokio::spawn(async move {
        if let Err(error) =
            transcript_writer_loop(session, utterance_rx, writer_runtime, transcript_jsonl_path)
                .await
        {
            tracing::error!("transcript writer stopped: {error:#}");
        }
    });

    attach_voice_handlers(
        state,
        VoiceHandlerAttachContext {
            http: Arc::clone(&ctx.http),
            guild_id,
            text_channel,
            voice_channel,
            call_lock: Arc::clone(&call_lock),
            runtime,
        },
    )
    .await;

    tokio::spawn(startup_receive_watchdog(
        ctx.clone(),
        Arc::clone(state),
        guild_id,
        voice_channel,
        text_channel,
        0,
    ));

    tokio::spawn(steady_state_receive_watchdog(
        ctx.clone(),
        Arc::clone(state),
        guild_id,
        voice_channel,
        text_channel,
    ));

    Ok(())
}

pub(crate) async fn maybe_load_thread_context(
    ctx: &Context,
    state: &Arc<AppState>,
    channel_id: ChannelId,
) -> anyhow::Result<()> {
    if state.transcript_threads.contains_key(&channel_id) {
        return Ok(());
    }

    let channel = channel_id.to_channel(&ctx.http).await?;
    let (parent_id, starter_message_id) = match channel {
        Channel::Guild(gc)
            if matches!(
                gc.kind,
                ChannelType::PublicThread | ChannelType::PrivateThread | ChannelType::NewsThread
            ) =>
        {
            let Some(parent_id) = gc.parent_id else {
                return Ok(());
            };
            (parent_id, MessageId::new(gc.id.get()))
        }
        _ => return Ok(()),
    };

    let starter = parent_id.message(&ctx.http, starter_message_id).await?;
    let Some(attachment) = starter
        .attachments
        .iter()
        .find(|a| a.filename.starts_with("transcript-") && a.filename.ends_with(".md"))
    else {
        return Ok(());
    };

    if attachment.size as u64 > TRANSCRIPT_ATTACHMENT_MAX_BYTES {
        tracing::warn!(
            thread = %channel_id,
            bytes = attachment.size,
            "skipping transcript attachment load: attachment exceeds size cap"
        );
        return Ok(());
    }

    let transcript_text = reqwest::get(&attachment.url).await?.text().await?;
    super::upsert_thread_context(state, channel_id, transcript_text);

    tracing::info!(thread = %channel_id, "loaded transcript context from attachment");
    Ok(())
}
