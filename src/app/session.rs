use std::collections::HashMap;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use serenity::all::{
    Channel, ChannelId, ChannelType, CreateAttachment, CreateMessage, GuildId, MessageId, UserId,
    VoiceState,
};
use serde::Deserialize;
use serenity::prelude::Context;
use songbird::events::{CoreEvent, Event};
use tokio::fs;

use super::{
    AppState, GuildRuntime, FINALIZE_SETTLE_PASSES, FINALIZE_SETTLE_TIMEOUT,
    STARTUP_RECEIVE_RECOVERY_MAX_ATTEMPTS, STARTUP_RECEIVE_WATCHDOG_DELAY,
    STEADY_STATE_NO_PROGRESS_TIMEOUT, STEADY_STATE_WATCHDOG_CADENCE, Utterance,
};
use crate::audio::{
    clear_unknown_ssrc_audio_for_guild, ClientDisconnectHandler, SpeakingUpdateHandler,
    VoiceTickHandler,
};
use crate::gemini::summarize_transcript;
use crate::transcription::{should_dispatch_chunk, transcribe_mono_pcm, trim_finalize_tail};

const TRANSCRIPT_ATTACHMENT_MAX_BYTES: u64 = 10 * 1024 * 1024;
const THREAD_SUMMARY_MAX_CHARS: usize = 1_800;

pub struct VoiceHandlerAttachContext {
    pub http: Arc<serenity::http::Http>,
    pub guild_id: GuildId,
    pub text_channel: ChannelId,
    pub voice_channel: ChannelId,
    pub call_lock: Arc<tokio::sync::Mutex<songbird::Call>>,
    pub runtime: Arc<GuildRuntime>,
}

pub async fn finalize_call_for_guild(
    ctx: &Context,
    state: &Arc<AppState>,
    guild_id: GuildId,
) -> anyhow::Result<()> {
    let Some((_gid, session_lock)) = state.active_calls.remove(&guild_id) else {
        return Ok(());
    };

    let manager = songbird::get(ctx)
        .await
        .context("songbird voice manager unavailable")?
        .clone();
    let _ = manager.remove(guild_id).await;

    settle_and_flush_guild_audio(state, guild_id).await;
    wait_for_transcription_drain(state, guild_id).await;
    wait_for_transcript_commit_drain(state, guild_id).await;

    for key in state
        .ssrc_to_user
        .iter()
        .map(|e| *e.key())
        .filter(|(g, _)| *g == guild_id)
        .collect::<Vec<_>>()
    {
        state.ssrc_to_user.remove(&key);
    }
    clear_unknown_ssrc_audio_for_guild(guild_id);

    for key in state
        .streams
        .iter()
        .map(|e| *e.key())
        .filter(|(g, _)| *g == guild_id)
        .collect::<Vec<_>>()
    {
        state.streams.remove(&key);
    }

    let session = session_lock.read().await;
    let transcript = load_persisted_transcript(
        &session.transcript_jsonl_path,
        session.started_mono,
    )
    .await;

    let transcript_title = format_call_title(session.started_at);
    let call_duration = session.started_mono.elapsed();
    let should_generate_summary = state.post_call_summary_enabled;
    let include_summary_in_markdown =
        should_generate_summary && state.post_call_summary_include_in_markdown;
    let mut auto_summary = if include_summary_in_markdown {
        maybe_generate_post_call_summary(
            ctx,
            state,
            &transcript,
            session.started_at,
        )
        .await
    } else {
        None
    };
    let transcript_text =
        format_export_markdown(
            ctx,
            &transcript,
            session.started_at,
            call_duration,
            &transcript_title,
            auto_summary.as_deref(),
            include_summary_in_markdown,
        )
        .await;
    let filename = format!("transcript-{}.md", session.started_at.format("%Y%m%d-%H%M%S"));

    let local_dir = PathBuf::from("transcripts");
    fs::create_dir_all(&local_dir)
        .await
        .context("failed to create local transcript directory")?;
    let local_path = local_dir.join(&filename);
    fs::write(&local_path, &transcript_text).await.with_context(|| {
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

    let thread = session
        .text_channel
        .create_thread_from_message(
            &ctx.http,
            msg.id,
            serenity::builder::CreateThread::new(transcript_title),
        )
        .await?;

    if should_generate_summary && state.post_call_summary_post_in_thread {
        if auto_summary.is_none() {
            auto_summary = maybe_generate_post_call_summary(
                ctx,
                state,
                &transcript,
                session.started_at,
            )
            .await;
        }

        if let Some(summary) = auto_summary.as_deref() {
            let summary_message = format_summary_thread_message(summary);
            if let Err(e) = thread.say(&ctx.http, summary_message).await {
                tracing::warn!(guild = %guild_id, "failed to post auto-summary in thread: {e:#}");
            }
        }
    }

    super::upsert_thread_context(state, thread.id, transcript_text);

    state.guild_runtimes.remove(&guild_id);

    Ok(())
}

#[derive(Deserialize)]
struct PersistedUtterance {
    user_id: u64,
    start_offset_ms: u64,
    text: String,
}

async fn load_persisted_transcript(path: &std::path::Path, started_mono: Instant) -> Vec<Utterance> {
    let Ok(content) = fs::read_to_string(path).await else {
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
        runtime,
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
            runtime,
            voice_channel,
            guild_id,
            ssrc_to_user: Arc::clone(&state.ssrc_to_user),
            streams: Arc::clone(&state.streams),
            enable_denoiser: state.enable_denoiser,
            asr: Arc::clone(&state.asr),
            live_transcript_debug: state.live_transcript_debug,
            silence_ticks_threshold: state.endpoint_silence_ticks,
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

    let runtime = if let Some(runtime) = state
        .guild_runtimes
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    {
        runtime
    } else {
        tracing::debug!(
            guild = %guild_id,
            attempt,
            "startup watchdog exiting: missing guild runtime"
        );
        return;
    };

    if !state.active_calls.contains_key(&guild_id) {
        tracing::debug!(
            guild = %guild_id,
            attempt,
            "startup watchdog exiting: no active call session"
        );
        return;
    }

    let activity = runtime.decoded_audio_activity.load(Ordering::SeqCst);
    if activity > 0 {
        tracing::info!(
            guild = %guild_id,
            attempt,
            decoded_frames = activity,
            "startup watchdog healthy: decoded audio observed"
        );
        return;
    }

    let decode_failures = runtime.decode_failure_activity.load(Ordering::SeqCst);
    let unmapped_ssrc = runtime.unmapped_ssrc_activity.load(Ordering::SeqCst);
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

    let Ok(_recovery_guard) = runtime.recovery_lock.try_lock() else {
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
    clear_unknown_ssrc_audio_for_guild(guild_id);

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

    runtime.decoded_audio_activity.store(0, Ordering::SeqCst);
    runtime.decode_failure_activity.store(0, Ordering::SeqCst);
    runtime.unmapped_ssrc_activity.store(0, Ordering::SeqCst);

    attach_voice_handlers(
        &state,
        VoiceHandlerAttachContext {
            http: Arc::clone(&ctx.http),
            guild_id,
            text_channel,
            voice_channel,
            call_lock: Arc::clone(&call_lock),
            runtime: Arc::clone(&runtime),
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
    let Some(runtime) = state
        .guild_runtimes
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        tracing::debug!(
            guild = %guild_id,
            "steady-state watchdog exiting: missing guild runtime"
        );
        return;
    };

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

        let decode_activity = runtime.decoded_audio_activity.load(Ordering::SeqCst);

        if decode_activity > last_decode_activity {
            last_decode_activity = decode_activity;
            last_progress = Instant::now();
            continue;
        }

        if last_progress.elapsed() < STEADY_STATE_NO_PROGRESS_TIMEOUT {
            continue;
        }

        let decode_failures = runtime.decode_failure_activity.load(Ordering::SeqCst);
        let unmapped_ssrc = runtime.unmapped_ssrc_activity.load(Ordering::SeqCst);

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
        last_decode_activity = runtime.decoded_audio_activity.load(Ordering::SeqCst);
    }
}

async fn settle_and_flush_guild_audio(
    state: &Arc<AppState>,
    guild_id: GuildId,
) {
    let Some(runtime) = state
        .guild_runtimes
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        return;
    };

    for _ in 0..FINALIZE_SETTLE_PASSES {
        let _ = wait_for_capture_quiesce_with_timeout(state, guild_id, FINALIZE_SETTLE_TIMEOUT).await;

        let pending = flush_pending_buffers_for_export(state, guild_id).await;
        if !pending.is_empty() {
            commit_flushed_utterances(state, guild_id, pending).await;
        }

        let inflight = runtime.transcription_inflight.load(Ordering::SeqCst);
        let pending_commits = runtime.transcript_pending_commits.load(Ordering::SeqCst);
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
        .guild_runtimes
        .get(&guild_id)
        .map(|v| Arc::clone(&v.transcription_inflight))
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
        let Some(runtime) = state
            .guild_runtimes
            .get(&guild_id)
            .map(|v| Arc::clone(v.value()))
        else {
            return true;
        };

        let inflight = runtime.transcription_inflight.load(Ordering::SeqCst);
        let pending_commits = runtime.transcript_pending_commits.load(Ordering::SeqCst);

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
        .guild_runtimes
        .get(&guild_id)
        .map(|v| Arc::clone(&v.transcript_pending_commits))
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
    let Some(runtime) = state
        .guild_runtimes
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        tracing::warn!(
            guild = %guild_id,
            flushed = pending.len(),
            "missing guild runtime; dropped flushed tail utterances during finalize"
        );
        return;
    };

    for utterance in pending {
        runtime
            .transcript_pending_commits
            .fetch_add(1, Ordering::SeqCst);
        if runtime.utterance_tx.send(utterance).await.is_err() {
            runtime
                .transcript_pending_commits
                .fetch_sub(1, Ordering::SeqCst);
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
        let mut voiced_ticks = 0u32;
        let mut noise_rms_ema = 0.0f32;

        if let Some(mut stream) = state.streams.get_mut(&user_key) {
            let flushed = stream.denoiser.flush_pending();
            noise_rms_ema = stream.denoiser.noise_rms_ema();
            let entry = &mut stream.buffer;
            entry.pcm.extend(flushed);

            if entry.pcm.is_empty() {
                continue;
            }

            start_ts = Some(entry.utterance_start.take().unwrap_or_else(Instant::now));
            pcm = std::mem::take(&mut entry.pcm);
            voiced_ticks = std::mem::take(&mut entry.voiced_ticks);
            trim_finalize_tail(&mut pcm, entry.silent_ticks);
            entry.silent_ticks = 0;
        }

        let Some(start_ts) = start_ts else {
            continue;
        };

        if let Err(rejection) = should_dispatch_chunk(&pcm, voiced_ticks, noise_rms_ema) {
            if let Some(runtime) = state.guild_runtimes.get(&guild_id) {
                runtime.dispatch_gate_total.fetch_add(1, Ordering::SeqCst);
            }
            tracing::debug!(
                guild = %guild_id,
                user = %user_id,
                stage = "finalize_flush",
                reason = rejection.reason,
                voiced_ticks = rejection.voiced_ticks,
                rms = rejection.rms,
                floor = rejection.floor,
                "dispatch gate rejected utterance"
            );
            continue;
        }

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
    let by_user = resolve_display_names(ctx, transcript).await;
    let mut lines = build_transcript_lines(transcript, &by_user);

    lines.insert(0, String::new());
    lines.insert(0, "## Transcript".to_string());
    lines.insert(0, String::new());
    lines.insert(
        0,
        format!(
            "**Started:** {} UTC",
            started_at.format("%Y-%m-%d %H:%M:%S")
        ),
    );
    lines.insert(0, String::new());
    lines.insert(0, "# Meeting Transcript".to_string());

    lines.join("\n")
}

async fn format_export_markdown(
    ctx: &Context,
    transcript: &[Utterance],
    started_at: chrono::DateTime<chrono::Utc>,
    call_duration: Duration,
    title: &str,
    summary: Option<&str>,
    include_summary_in_markdown: bool,
) -> String {
    let by_user = resolve_display_names(ctx, transcript).await;
    let attendees = attendees_in_order(transcript, &by_user);
    let duration = format_duration(call_duration);

    let mut out = Vec::new();
    out.push("---".to_string());
    out.push(format!("title: \"{}\"", yaml_escape_double_quoted(title)));
    out.push("type: meeting".to_string());
    out.push(format!(
        "date: {}",
        started_at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
    ));
    out.push(format!("duration: \"{}\"", duration));
    out.push("source: discord".to_string());
    out.push("status: complete".to_string());
    if !attendees.is_empty() {
        out.push("attendees:".to_string());
        for attendee in attendees {
            out.push(format!("  - {}", yaml_single_line_scalar(&attendee)));
        }
    }
    out.push("---".to_string());
    out.push(String::new());
    if include_summary_in_markdown {
        if let Some(summary_text) = summary.map(str::trim).filter(|s| !s.is_empty()) {
            out.push(summary_text.to_string());
            out.push(String::new());
        }
    }
    out.push("## Transcript".to_string());
    out.push(String::new());
    out.extend(build_transcript_lines(transcript, &by_user));
    out.push(String::new());

    out.join("\n")
}

fn format_call_title(started_at: chrono::DateTime<chrono::Utc>) -> String {
    format!("Transcript {}", started_at.format("%Y-%m-%d %H:%M:%S UTC"))
}

async fn maybe_generate_post_call_summary(
    ctx: &Context,
    state: &Arc<AppState>,
    transcript: &[Utterance],
    started_at: chrono::DateTime<chrono::Utc>,
) -> Option<String> {
    if !state.post_call_summary_enabled {
        return None;
    }

    if transcript.is_empty() {
        return None;
    }

    let transcript_context = format_transcript(ctx, transcript, started_at).await;
    let timeout = Duration::from_secs(state.post_call_summary_timeout_secs.max(5));

    match tokio::time::timeout(
        timeout,
        summarize_transcript(&state.gemini_key, &state.gemini_model, &transcript_context),
    )
    .await
    {
        Ok(Ok(summary)) => {
            let summary = summary.trim().to_string();
            if summary.is_empty() {
                tracing::warn!("auto-summary returned empty text");
                None
            } else {
                Some(summary)
            }
        }
        Ok(Err(e)) => {
            tracing::warn!("auto-summary failed: {e:#}");
            None
        }
        Err(_) => {
            tracing::warn!(timeout_secs = state.post_call_summary_timeout_secs, "auto-summary timed out");
            None
        }
    }
}

fn format_summary_thread_message(summary: &str) -> String {
    let trimmed = summary.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut message = trimmed.to_string();
    if message.chars().count() > THREAD_SUMMARY_MAX_CHARS {
        let keep = THREAD_SUMMARY_MAX_CHARS.saturating_sub(18);
        let truncated: String = message.chars().take(keep).collect();
        message = format!("{truncated}\n\n(truncated)");
    }
    message
}

async fn resolve_display_names(ctx: &Context, transcript: &[Utterance]) -> HashMap<UserId, String> {
    let mut by_user = HashMap::<UserId, String>::new();
    for utt in transcript {
        if by_user.contains_key(&utt.user_id) {
            continue;
        }
        let name = match utt.user_id.to_user(&ctx.http).await {
            Ok(u) => u.display_name().to_string(),
            Err(_) => format!("{}", utt.user_id.get()),
        };
        by_user.insert(utt.user_id, name);
    }
    by_user
}

fn attendees_in_order(transcript: &[Utterance], by_user: &HashMap<UserId, String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut attendees = Vec::new();
    for utt in transcript {
        if !seen.insert(utt.user_id) {
            continue;
        }
        if let Some(name) = by_user.get(&utt.user_id) {
            attendees.push(name.clone());
        }
    }
    attendees
}

fn build_transcript_lines(transcript: &[Utterance], by_user: &HashMap<UserId, String>) -> Vec<String> {
    if transcript.is_empty() {
        return vec!["_No captured speech._".to_string()];
    }

    let mut lines = Vec::with_capacity(transcript.len());
    let first = transcript[0].start_ts;

    for utt in transcript {
        let display = by_user
            .get(&utt.user_id)
            .cloned()
            .unwrap_or_else(|| format!("{}", utt.user_id.get()));

        let delta = utt.start_ts.saturating_duration_since(first);
        let stamp = format_transcript_stamp(delta);
        lines.push(format!("[{display} {stamp}] {}", utt.text));
    }

    lines
}

fn format_transcript_stamp(delta: Duration) -> String {
    let total = delta.as_secs();
    let mm = total / 60;
    let ss = total % 60;
    format!("{mm}:{ss:02}")
}

fn format_duration(duration: Duration) -> String {
    let total = duration.as_secs();
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;

    if hours > 0 {
        format!("{hours}h {minutes}m {seconds}s")
    } else {
        format!("{minutes}m {seconds}s")
    }
}

fn yaml_escape_double_quoted(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn yaml_single_line_scalar(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '-' | '_' | '.'))
    {
        value.to_string()
    } else {
        format!("\"{}\"", yaml_escape_double_quoted(value))
    }
}

pub(super) async fn maybe_load_thread_context(
    ctx: &Context,
    state: &Arc<AppState>,
    channel_id: ChannelId,
) -> anyhow::Result<()> {
    ensure_thread_context_loaded(ctx, state, channel_id).await
}
