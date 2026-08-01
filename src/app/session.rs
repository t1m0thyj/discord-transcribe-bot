use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use serenity::all::{
    Channel, ChannelId, ChannelType, CreateAttachment, CreateMessage, GuildId, MessageId, UserId,
    VoiceState,
};
use serde::Deserialize;
use serenity::prelude::Context;
use songbird::events::{CoreEvent, Event};
use tokio::sync::mpsc;

use super::{
    AppState, FINALIZE_SETTLE_PASSES, FINALIZE_SETTLE_TIMEOUT,
    STARTUP_RECEIVE_RECOVERY_MAX_ATTEMPTS, STARTUP_RECEIVE_WATCHDOG_DELAY,
    STEADY_STATE_NO_PROGRESS_TIMEOUT, STEADY_STATE_WATCHDOG_CADENCE, ThreadContext, Utterance,
};
use crate::audio::{ClientDisconnectHandler, SpeakingUpdateHandler, VoiceTickHandler};
use crate::transcription::transcribe_mono_pcm;

pub struct VoiceHandlerAttachContext {
    pub http: Arc<serenity::http::Http>,
    pub guild_id: GuildId,
    pub text_channel: ChannelId,
    pub voice_channel: ChannelId,
    pub call_lock: Arc<tokio::sync::Mutex<songbird::Call>>,
    pub utterance_tx: mpsc::Sender<Utterance>,
    pub inflight: Arc<AtomicUsize>,
    pub pending_commits: Arc<AtomicUsize>,
    pub decode_shed_total: Arc<AtomicUsize>,
    pub decode_activity: Arc<AtomicUsize>,
    pub decode_failure_activity: Arc<AtomicUsize>,
    pub unmapped_ssrc_activity: Arc<AtomicUsize>,
    pub started_notified: Arc<AtomicBool>,
}

pub async fn finalize_call_for_guild(
    ctx: &Context,
    state: &Arc<AppState>,
    guild_id: GuildId,
) -> anyhow::Result<()> {
    let Some((_gid, session_lock)) = state.active_calls.remove(&guild_id) else {
        return Ok(());
    };

    settle_and_flush_guild_audio(state, guild_id).await;
    wait_for_transcription_drain(state, guild_id).await;
    wait_for_transcript_commit_drain(state, guild_id).await;

    let manager = songbird::get(ctx)
        .await
        .context("songbird voice manager unavailable")?
        .clone();
    let _ = manager.remove(guild_id).await;

    for key in state
        .ssrc_to_user
        .iter()
        .map(|e| *e.key())
        .filter(|(g, _)| *g == guild_id)
        .collect::<Vec<_>>()
    {
        state.ssrc_to_user.remove(&key);
    }

    for key in state
        .streams
        .iter()
        .map(|e| *e.key())
        .filter(|(g, _)| *g == guild_id)
        .collect::<Vec<_>>()
    {
        state.streams.remove(&key);
    }

    settle_and_flush_guild_audio(state, guild_id).await;
    wait_for_transcription_drain(state, guild_id).await;
    wait_for_transcript_commit_drain(state, guild_id).await;

    state.utterance_senders.remove(&guild_id);

    let session = session_lock.read().await;
    let transcript = load_persisted_transcript(
        &session.transcript_jsonl_path,
        session.started_mono,
    );

    let transcript_text = format_transcript(ctx, &transcript, session.started_at).await;
    let filename = format!(
        "transcript-{}-{}.txt",
        guild_id.get(),
        session.started_at.format("%Y%m%d-%H%M%S")
    );

    let local_dir = PathBuf::from("transcripts");
    fs::create_dir_all(&local_dir)
        .context("failed to create local transcript directory")?;
    let local_path = local_dir.join(&filename);
    fs::write(&local_path, &transcript_text).with_context(|| {
        format!(
            "failed to write local transcript file {}",
            local_path.display()
        )
    })?;

    let attachment =
        CreateAttachment::bytes(transcript_text.clone().into_bytes(), filename.clone());
    let msg = session
        .text_channel
        .send_files(
            &ctx.http,
            vec![attachment],
            CreateMessage::new().content(
                "Call transcript attached. Ask AI questions in the thread on this message.",
            ),
        )
        .await?;

    let thread_name = format!(
        "Transcript {}",
        session.started_at.format("%Y-%m-%d %H:%M:%S UTC")
    );

    let thread = session
        .text_channel
        .create_thread_from_message(
            &ctx.http,
            msg.id,
            serenity::builder::CreateThread::new(thread_name),
        )
        .await?;

    state.transcript_threads.insert(
        thread.id,
        ThreadContext {
            transcript: transcript_text,
            history: Vec::new(),
        },
    );

    state.transcription_inflight.remove(&guild_id);
    state.transcript_pending_commits.remove(&guild_id);
    state.decode_shed_total.remove(&guild_id);
    state.recovery_locks.remove(&guild_id);
    state.decoded_audio_activity.remove(&guild_id);
    state.decode_failure_activity.remove(&guild_id);
    state.unmapped_ssrc_activity.remove(&guild_id);
    state.transcription_started_notified.remove(&guild_id);

    Ok(())
}

#[derive(Deserialize)]
struct PersistedUtterance {
    user_id: u64,
    start_offset_ms: u64,
    text: String,
}

fn load_persisted_transcript(path: &std::path::Path, started_mono: Instant) -> Vec<Utterance> {
    let Ok(content) = fs::read_to_string(path) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for line in content.lines() {
        let Ok(item) = serde_json::from_str::<PersistedUtterance>(line) else {
            continue;
        };
        out.push(Utterance {
            user_id: UserId::new(item.user_id),
            start_ts: started_mono + Duration::from_millis(item.start_offset_ms),
            text: item.text,
        });
    }

    out.sort_by_key(|u| u.start_ts);

    out
}

pub async fn attach_voice_handlers(state: &Arc<AppState>, ctx: VoiceHandlerAttachContext) {
    let VoiceHandlerAttachContext {
        http,
        guild_id,
        text_channel,
        voice_channel,
        call_lock,
        utterance_tx,
        inflight,
        pending_commits,
        decode_shed_total,
        decode_activity,
        decode_failure_activity,
        unmapped_ssrc_activity,
        started_notified,
    } = ctx;

    let mut call = call_lock.lock().await;
    call.add_global_event(
        Event::Core(CoreEvent::SpeakingStateUpdate),
        SpeakingUpdateHandler {
            guild_id,
            ssrc_to_user: Arc::clone(&state.ssrc_to_user),
        },
    );
    call.add_global_event(
        Event::Core(CoreEvent::ClientDisconnect),
        ClientDisconnectHandler {
            guild_id,
            ssrc_to_user: Arc::clone(&state.ssrc_to_user),
        },
    );
    call.add_global_event(
        Event::Core(CoreEvent::VoiceTick),
        VoiceTickHandler {
            http,
            text_channel,
            started_notified,
            voice_channel,
            guild_id,
            ssrc_to_user: Arc::clone(&state.ssrc_to_user),
            streams: Arc::clone(&state.streams),
            enable_denoiser: state.enable_denoiser,
            utterance_tx,
            transcription_inflight: inflight,
            transcript_pending_commits: pending_commits,
            decode_shed_total,
            decode_activity,
            decode_failure_activity,
            unmapped_ssrc_activity,
            asr: Arc::clone(&state.asr),
            live_transcript_debug: state.live_transcript_debug,
            rolling_ingest_max_ms: state.rolling_ingest_max_ms,
            rolling_ingest_context_ms: state.rolling_ingest_context_ms,
        },
    );
}

pub async fn startup_receive_watchdog(
    ctx: Context,
    state: Arc<AppState>,
    guild_id: GuildId,
    voice_channel: ChannelId,
    text_channel: ChannelId,
    attempt: u8,
) {
    async fn schedule_next_retry(
        ctx: Context,
        state: Arc<AppState>,
        guild_id: GuildId,
        voice_channel: ChannelId,
        text_channel: ChannelId,
        attempt: u8,
    ) {
        Box::pin(startup_receive_watchdog(
            ctx,
            state,
            guild_id,
            voice_channel,
            text_channel,
            attempt.saturating_add(1),
        ))
        .await;
    }

    tokio::time::sleep(STARTUP_RECEIVE_WATCHDOG_DELAY).await;

    if !state.active_calls.contains_key(&guild_id) {
        tracing::debug!(
            guild = %guild_id,
            attempt,
            "startup watchdog exiting: no active call session"
        );
        return;
    }

    let activity = state
        .decoded_audio_activity
        .get(&guild_id)
        .map(|v| v.load(Ordering::SeqCst))
        .unwrap_or(0);
    if activity > 0 {
        tracing::info!(
            guild = %guild_id,
            attempt,
            decoded_frames = activity,
            "startup watchdog healthy: decoded audio observed"
        );
        return;
    }

    let decode_failures = state
        .decode_failure_activity
        .get(&guild_id)
        .map(|v| v.load(Ordering::SeqCst))
        .unwrap_or(0);
    let unmapped_ssrc = state
        .unmapped_ssrc_activity
        .get(&guild_id)
        .map(|v| v.load(Ordering::SeqCst))
        .unwrap_or(0);
    let receive_errors = decode_failures.saturating_add(unmapped_ssrc);
    // First recovery attempt must be justified by explicit decode/mapping failures.
    // After recovery has started, continue bounded retries even if fresh decode errors
    // are not observed, since broken receive can produce zero packets/errors.
    if receive_errors == 0 && attempt == 0 {
        tracing::debug!(
            guild = %guild_id,
            attempt,
            "startup watchdog idle: no decode/mapping failures observed yet"
        );
        return;
    }

    let non_bot_present = match ctx.cache.guild(guild_id) {
        Some(guild) => {
            let bot_id = ctx.cache.current_user().id;
            guild
                .voice_states
                .iter()
                .any(|(uid, vs)| vs.channel_id == Some(voice_channel) && *uid != bot_id)
        }
        None => false,
    };
    if !non_bot_present {
        tracing::info!(
            guild = %guild_id,
            attempt,
            "startup watchdog exiting: no non-bot users in voice channel"
        );
        return;
    }

    if attempt >= STARTUP_RECEIVE_RECOVERY_MAX_ATTEMPTS {
        tracing::error!(
            guild = %guild_id,
            "startup receive remained unhealthy after recovery attempts; finalizing session for clean reset"
        );
        let _ = text_channel
            .say(
                &ctx.http,
                "Transcription receive stayed unhealthy after multiple retries. Resetting this session so you can /join again cleanly.",
            )
            .await;
        let _ = finalize_call_for_guild(&ctx, &state, guild_id).await;
        return;
    }

    tracing::warn!(
        guild = %guild_id,
        attempt,
        decode_failures,
        unmapped_ssrc,
        "decode/mapping failures observed without usable audio after join; reinitializing voice receive"
    );

    let recovery_lock = if let Some(lock) = state.recovery_locks.get(&guild_id) {
        Arc::clone(lock.value())
    } else {
        let lock = Arc::new(tokio::sync::Mutex::new(()));
        state.recovery_locks.insert(guild_id, Arc::clone(&lock));
        lock
    };
    let Ok(_recovery_guard) = recovery_lock.try_lock() else {
        tracing::debug!(
            guild = %guild_id,
            attempt,
            "startup recovery skipped: another recovery attempt is already in progress"
        );
        return;
    };

    let _ = text_channel
        .say(
            &ctx.http,
            format!(
                "Audio decode/mapping failures observed after join; reinitializing voice receive (attempt {}/{})...",
                attempt + 1,
                STARTUP_RECEIVE_RECOVERY_MAX_ATTEMPTS
            ),
        )
        .await;

    let manager = match songbird::get(&ctx).await {
        Some(m) => m.clone(),
        None => {
            tracing::warn!(
                guild = %guild_id,
                attempt,
                "startup recovery retrying: songbird manager unavailable"
            );
            schedule_next_retry(ctx, state, guild_id, voice_channel, text_channel, attempt).await;
            return;
        }
    };

    let _ = manager.remove(guild_id).await;

    for key in state
        .ssrc_to_user
        .iter()
        .map(|e| *e.key())
        .filter(|(g, _)| *g == guild_id)
        .collect::<Vec<_>>()
    {
        state.ssrc_to_user.remove(&key);
    }

    let user_keys: HashSet<(GuildId, UserId)> = state
        .streams
        .iter()
        .map(|e| *e.key())
        .filter(|(g, _)| *g == guild_id)
        .collect();
    for key in user_keys.iter() {
        state.streams.remove(key);
    }

    let Ok(call_lock) = manager.join(guild_id, voice_channel).await else {
        tracing::warn!(
            guild = %guild_id,
            attempt,
            "startup recovery retrying: failed to rejoin voice channel"
        );
        schedule_next_retry(ctx, state, guild_id, voice_channel, text_channel, attempt).await;
        return;
    };

    let Some(utterance_tx) = state
        .utterance_senders
        .get(&guild_id)
        .map(|v| v.value().clone())
    else {
        tracing::warn!(
            guild = %guild_id,
            attempt,
            "startup recovery retrying: missing utterance sender state"
        );
        schedule_next_retry(ctx, state, guild_id, voice_channel, text_channel, attempt).await;
        return;
    };
    let Some(inflight) = state
        .transcription_inflight
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        tracing::warn!(
            guild = %guild_id,
            attempt,
            "startup recovery retrying: missing inflight counter state"
        );
        schedule_next_retry(ctx, state, guild_id, voice_channel, text_channel, attempt).await;
        return;
    };
    let Some(pending_commits) = state
        .transcript_pending_commits
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        tracing::warn!(
            guild = %guild_id,
            attempt,
            "startup recovery retrying: missing pending commits counter state"
        );
        schedule_next_retry(ctx, state, guild_id, voice_channel, text_channel, attempt).await;
        return;
    };
    let Some(decode_activity) = state
        .decoded_audio_activity
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        tracing::warn!(
            guild = %guild_id,
            attempt,
            "startup recovery retrying: missing decode activity state"
        );
        schedule_next_retry(ctx, state, guild_id, voice_channel, text_channel, attempt).await;
        return;
    };
    let Some(decode_failure_activity) = state
        .decode_failure_activity
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        tracing::warn!(
            guild = %guild_id,
            attempt,
            "startup recovery retrying: missing decode-failure counter state"
        );
        schedule_next_retry(ctx, state, guild_id, voice_channel, text_channel, attempt).await;
        return;
    };
    let Some(unmapped_ssrc_activity) = state
        .unmapped_ssrc_activity
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        tracing::warn!(
            guild = %guild_id,
            attempt,
            "startup recovery retrying: missing unmapped-ssrc counter state"
        );
        schedule_next_retry(ctx, state, guild_id, voice_channel, text_channel, attempt).await;
        return;
    };
    let Some(decode_shed_total) = state
        .decode_shed_total
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        tracing::warn!(
            guild = %guild_id,
            attempt,
            "startup recovery retrying: missing decode shed counter state"
        );
        schedule_next_retry(ctx, state, guild_id, voice_channel, text_channel, attempt).await;
        return;
    };
    let Some(started_notified) = state
        .transcription_started_notified
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        tracing::warn!(
            guild = %guild_id,
            attempt,
            "startup recovery retrying: missing started-notified state"
        );
        schedule_next_retry(ctx, state, guild_id, voice_channel, text_channel, attempt).await;
        return;
    };
    decode_activity.store(0, Ordering::SeqCst);
    decode_failure_activity.store(0, Ordering::SeqCst);
    unmapped_ssrc_activity.store(0, Ordering::SeqCst);

    attach_voice_handlers(
        &state,
        VoiceHandlerAttachContext {
            http: Arc::clone(&ctx.http),
            guild_id,
            text_channel,
            voice_channel,
            call_lock: Arc::clone(&call_lock),
            utterance_tx,
            inflight,
            pending_commits,
            decode_shed_total,
            decode_activity,
            decode_failure_activity,
            unmapped_ssrc_activity,
            started_notified,
        },
    )
    .await;

    tracing::info!(
        guild = %guild_id,
        attempt,
        "startup recovery reattached voice handlers; waiting for healthy decoded audio"
    );

    schedule_next_retry(ctx, state, guild_id, voice_channel, text_channel, attempt).await;
}

pub async fn steady_state_receive_watchdog(
    ctx: Context,
    state: Arc<AppState>,
    guild_id: GuildId,
    voice_channel: ChannelId,
    text_channel: ChannelId,
) {
    let mut last_decode_activity = 0usize;
    let mut last_progress = Instant::now();

    loop {
        tokio::time::sleep(STEADY_STATE_WATCHDOG_CADENCE).await;

        if !state.active_calls.contains_key(&guild_id) {
            tracing::debug!(
                guild = %guild_id,
                "steady-state watchdog exiting: no active call session"
            );
            return;
        }

        let non_bot_present = match ctx.cache.guild(guild_id) {
            Some(guild) => {
                let bot_id = ctx.cache.current_user().id;
                guild
                    .voice_states
                    .iter()
                    .any(|(uid, vs)| vs.channel_id == Some(voice_channel) && *uid != bot_id)
            }
            None => false,
        };

        if !non_bot_present {
            last_progress = Instant::now();
            continue;
        }

        let decode_activity = state
            .decoded_audio_activity
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(0);

        if decode_activity > last_decode_activity {
            last_decode_activity = decode_activity;
            last_progress = Instant::now();
            continue;
        }

        if last_progress.elapsed() < STEADY_STATE_NO_PROGRESS_TIMEOUT {
            continue;
        }

        let decode_failures = state
            .decode_failure_activity
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(0);
        let unmapped_ssrc = state
            .unmapped_ssrc_activity
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(0);

        tracing::warn!(
            guild = %guild_id,
            decode_activity,
            decode_failures,
            unmapped_ssrc,
            "steady-state watchdog detected stalled receive path; forcing startup-style recovery"
        );
        let _ = text_channel
            .say(
                &ctx.http,
                "No decoded audio has arrived for over 60s while users are still in voice; reinitializing voice receive.",
            )
            .await;

        startup_receive_watchdog(
            ctx.clone(),
            Arc::clone(&state),
            guild_id,
            voice_channel,
            text_channel,
            1,
        )
        .await;

        last_progress = Instant::now();
        last_decode_activity = state
            .decoded_audio_activity
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(0);
    }
}

async fn settle_and_flush_guild_audio(
    state: &Arc<AppState>,
    guild_id: GuildId,
) {
    for _ in 0..FINALIZE_SETTLE_PASSES {
        let _ = wait_for_capture_quiesce_with_timeout(state, guild_id, FINALIZE_SETTLE_TIMEOUT).await;

        let pending = flush_pending_buffers_for_export(state, guild_id).await;
        if !pending.is_empty() {
            commit_flushed_utterances(state, guild_id, pending).await;
        }

        let inflight = state
            .transcription_inflight
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(0);
        let pending_commits = state
            .transcript_pending_commits
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(0);
        let buffered_audio = state
            .streams
            .iter()
            .any(|e| e.key().0 == guild_id && !e.value().buffer.pcm.is_empty());

        if inflight == 0 && pending_commits == 0 && !buffered_audio {
            break;
        }
    }
}

async fn wait_for_transcription_drain(state: &Arc<AppState>, guild_id: GuildId) {
    let Some(counter) = state
        .transcription_inflight
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        return;
    };

    let drained = tokio::time::timeout(Duration::from_secs(30), async {
        while counter.load(Ordering::SeqCst) > 0 {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await;

    if drained.is_err() {
        tracing::warn!(
            guild = %guild_id,
            "timed out waiting for transcription drain; continuing finalize with partial state"
        );
    }
}

async fn wait_for_capture_quiesce_with_timeout(
    state: &Arc<AppState>,
    guild_id: GuildId,
    timeout: Duration,
) -> bool {
    let start = Instant::now();

    loop {
        let inflight = state
            .transcription_inflight
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(0);
        let pending_commits = state
            .transcript_pending_commits
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(0);

        if inflight == 0 && pending_commits == 0 {
            return true;
        }

        if start.elapsed() >= timeout {
            return false;
        }

        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn wait_for_transcript_commit_drain(state: &Arc<AppState>, guild_id: GuildId) {
    let Some(counter) = state
        .transcript_pending_commits
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        return;
    };

    let drained = tokio::time::timeout(Duration::from_secs(30), async {
        while counter.load(Ordering::SeqCst) > 0 {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await;

    if drained.is_err() {
        tracing::warn!(
            guild = %guild_id,
            "timed out waiting for transcript commit drain; continuing finalize with partial state"
        );
    }
}

async fn commit_flushed_utterances(state: &Arc<AppState>, guild_id: GuildId, pending: Vec<Utterance>) {
    let Some(tx) = state
        .utterance_senders
        .get(&guild_id)
        .map(|v| v.value().clone())
    else {
        tracing::warn!(
            guild = %guild_id,
            flushed = pending.len(),
            "missing utterance sender; dropped flushed tail utterances during finalize"
        );
        return;
    };

    let Some(pending_commits) = state
        .transcript_pending_commits
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        tracing::warn!(
            guild = %guild_id,
            flushed = pending.len(),
            "missing pending commits counter; dropped flushed tail utterances during finalize"
        );
        return;
    };

    for utterance in pending {
        pending_commits.fetch_add(1, Ordering::SeqCst);
        if tx.send(utterance).await.is_err() {
            pending_commits.fetch_sub(1, Ordering::SeqCst);
            tracing::warn!(
                guild = %guild_id,
                "failed to enqueue flushed tail utterance for journal commit"
            );
            break;
        }
    }
}

pub(super) async fn flush_pending_buffers_for_export(
    state: &Arc<AppState>,
    guild_id: GuildId,
) -> Vec<Utterance> {
    let user_keys: Vec<(GuildId, UserId)> = state
        .streams
        .iter()
        .map(|e| *e.key())
        .filter(|(g, _)| *g == guild_id)
        .collect();
    let mut out = Vec::new();

    for user_key in user_keys {
        let user_id = user_key.1;
        let mut start_ts = None;
        let mut pcm = Vec::new();

        if let Some(mut stream) = state.streams.get_mut(&user_key) {
            let flushed = stream.denoiser.flush_pending();
            let entry = &mut stream.buffer;
            entry.pcm.extend(flushed);

            if entry.pcm.is_empty() {
                continue;
            }

            start_ts = Some(entry.utterance_start.take().unwrap_or_else(Instant::now));
            pcm = std::mem::take(&mut entry.pcm);
            entry.silent_ticks = 0;
        }

        let Some(start_ts) = start_ts else {
            continue;
        };

        if let Some(text) = transcribe_finalized_mono_pcm(state, pcm).await {
            out.push(Utterance {
                user_id,
                start_ts,
                text,
            });
        }
    }

    out
}
async fn transcribe_finalized_mono_pcm(state: &Arc<AppState>, pcm: Vec<f32>) -> Option<String> {
    transcribe_mono_pcm(Arc::clone(&state.asr), pcm).await
}

async fn ensure_thread_context_loaded(
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
        .find(|a| a.filename.starts_with("transcript-") && a.filename.ends_with(".txt"))
    else {
        return Ok(());
    };

    let transcript_text = reqwest::get(&attachment.url).await?.text().await?;
    state.transcript_threads.insert(
        channel_id,
        ThreadContext {
            transcript: transcript_text,
            history: Vec::new(),
        },
    );

    tracing::info!(thread = %channel_id, "loaded transcript context from attachment");
    Ok(())
}

pub async fn maybe_finalize_on_empty_voice_channel(
    ctx: &Context,
    state: &Arc<AppState>,
    guild_id: GuildId,
    old: Option<VoiceState>,
    new: VoiceState,
) -> anyhow::Result<()> {
    let Some(session_lock) = state.active_calls.get(&guild_id).map(|s| s.clone()) else {
        return Ok(());
    };

    let session = session_lock.read().await;
    let target_channel = session.voice_channel;
    let started_mono = session.started_mono;
    drop(session);

    let touched_target = new.channel_id == Some(target_channel)
        || old
            .as_ref()
            .and_then(|v| v.channel_id)
            .is_some_and(|c| c == target_channel);
    if !touched_target {
        return Ok(());
    }

    let bot_id = ctx.cache.current_user().id;
    let non_bot_user_departed_target = new.user_id != bot_id
        && old
            .as_ref()
            .and_then(|v| v.channel_id)
            .is_some_and(|c| c == target_channel)
        && new.channel_id != Some(target_channel);

    if started_mono.elapsed() < Duration::from_secs(10) && !non_bot_user_departed_target {
        return Ok(());
    }

    let non_bot_present = {
        let Some(guild) = ctx.cache.guild(guild_id) else {
            return Ok(());
        };

        guild
            .voice_states
            .iter()
            .any(|(uid, vs)| vs.channel_id == Some(target_channel) && *uid != bot_id)
    };

    if non_bot_present {
        return Ok(());
    }

    finalize_call_for_guild(ctx, state, guild_id).await?;

    Ok(())
}

pub async fn format_transcript(
    ctx: &Context,
    transcript: &[Utterance],
    started_at: chrono::DateTime<chrono::Utc>,
) -> String {
    if transcript.is_empty() {
        return format!(
            "Meeting transcript\nStarted: {} UTC\n\n(no captured speech)",
            started_at.format("%Y-%m-%d %H:%M:%S")
        );
    }

    let mut by_user = HashMap::<UserId, String>::new();
    let mut lines = Vec::with_capacity(transcript.len());
    let first = transcript[0].start_ts;

    lines.push("Meeting transcript".to_string());
    lines.push(format!(
        "Started: {} UTC",
        started_at.format("%Y-%m-%d %H:%M:%S")
    ));
    lines.push("Format: [HH:MM:SS] Speaker: text".to_string());
    lines.push(String::new());

    for utt in transcript {
        let display = if let Some(name) = by_user.get(&utt.user_id) {
            name.clone()
        } else {
            let name = match utt.user_id.to_user(&ctx.http).await {
                Ok(u) => u.display_name().to_string(),
                Err(_) => format!("{}", utt.user_id.get()),
            };
            by_user.insert(utt.user_id, name.clone());
            name
        };

        let delta = utt.start_ts.saturating_duration_since(first);
        let total = delta.as_secs();
        let hh = total / 3600;
        let mm = (total % 3600) / 60;
        let ss = total % 60;

        lines.push(format!("[{hh:02}:{mm:02}:{ss:02}] {display}: {}", utt.text));
    }

    lines.join("\n")
}

pub(super) async fn maybe_load_thread_context(
    ctx: &Context,
    state: &Arc<AppState>,
    channel_id: ChannelId,
) -> anyhow::Result<()> {
    ensure_thread_context_loaded(ctx, state, channel_id).await
}
