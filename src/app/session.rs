use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use serenity::all::{
    Channel, ChannelId, ChannelType, CreateAttachment, CreateMessage, GuildId, MessageId, UserId,
    VoiceState,
};
use serenity::prelude::Context;
use songbird::events::{CoreEvent, Event};
use tokio::sync::{RwLock, mpsc};

use super::{
    AppState, CallSession, FINALIZE_SETTLE_PASSES, FINALIZE_SETTLE_TIMEOUT,
    STARTUP_RECEIVE_RECOVERY_MAX_ATTEMPTS, STARTUP_RECEIVE_WATCHDOG_DELAY, ThreadContext,
    Utterance,
};
use crate::audio::{SpeakingUpdateHandler, VoiceTickHandler};
use crate::transcription::{make_revision_id, transcribe_mono_pcm};

pub struct VoiceHandlerAttachContext {
    pub http: Arc<serenity::http::Http>,
    pub guild_id: GuildId,
    pub text_channel: ChannelId,
    pub voice_channel: ChannelId,
    pub call_lock: Arc<tokio::sync::Mutex<songbird::Call>>,
    pub utterance_tx: mpsc::Sender<Utterance>,
    pub inflight: Arc<AtomicUsize>,
    pub pending_commits: Arc<AtomicUsize>,
    pub decode_activity: Arc<AtomicUsize>,
    pub decode_error_activity: Arc<AtomicUsize>,
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

    settle_and_flush_guild_audio(state, guild_id, &session_lock).await;
    wait_for_transcription_drain(state, guild_id).await;
    wait_for_transcript_commit_drain(state, guild_id).await;

    let manager = songbird::get(ctx)
        .await
        .context("songbird voice manager unavailable")?
        .clone();
    let _ = manager.remove(guild_id).await;

    settle_and_flush_guild_audio(state, guild_id, &session_lock).await;
    wait_for_transcription_drain(state, guild_id).await;
    wait_for_transcript_commit_drain(state, guild_id).await;

    state.utterance_senders.remove(&guild_id);

    let session = session_lock.read().await;
    let transcript = session.transcript.clone();

    let transcript_text = format_transcript(ctx, &transcript, session.started_at).await;
    let filename = format!(
        "transcript-{}-{}.txt",
        guild_id.get(),
        session.started_at.format("%Y%m%d-%H%M%S")
    );

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
    state.decoded_audio_activity.remove(&guild_id);
    state.decode_error_activity.remove(&guild_id);
    state.transcription_started_notified.remove(&guild_id);

    Ok(())
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
        decode_activity,
        decode_error_activity,
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
        Event::Core(CoreEvent::VoiceTick),
        VoiceTickHandler {
            http,
            text_channel,
            started_notified,
            voice_channel,
            guild_id,
            ssrc_to_user: Arc::clone(&state.ssrc_to_user),
            buffers: Arc::clone(&state.buffers),
            denoisers: Arc::clone(&state.denoisers),
            enable_denoiser: state.enable_denoiser,
            utterance_tx,
            transcription_inflight: inflight,
            transcript_pending_commits: pending_commits,
            decode_activity,
            decode_error_activity,
            asr: Arc::clone(&state.asr),
            asr_finalize: state.final_asr.as_ref().map(Arc::clone),
            live_transcript_debug: state.live_transcript_debug,
            provisional_step_ms: state.provisional_step_ms,
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

    let decode_errors = state
        .decode_error_activity
        .get(&guild_id)
        .map(|v| v.load(Ordering::SeqCst))
        .unwrap_or(0);
    // First recovery attempt must be justified by explicit decode/mapping failures.
    // After recovery has started, continue bounded retries even if fresh decode errors
    // are not observed, since broken receive can produce zero packets/errors.
    if decode_errors == 0 && attempt == 0 {
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
        decode_errors,
        "decode/mapping failures observed without usable audio after join; reinitializing voice receive"
    );

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
        .buffers
        .iter()
        .map(|e| *e.key())
        .filter(|(g, _)| *g == guild_id)
        .collect();
    for key in user_keys.iter() {
        state.buffers.remove(key);
        state.denoisers.remove(key);
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
    let Some(decode_error_activity) = state
        .decode_error_activity
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        tracing::warn!(
            guild = %guild_id,
            attempt,
            "startup recovery retrying: missing decode error state"
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
    decode_error_activity.store(0, Ordering::SeqCst);

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
            decode_activity,
            decode_error_activity,
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

async fn settle_and_flush_guild_audio(
    state: &Arc<AppState>,
    guild_id: GuildId,
    session_lock: &Arc<RwLock<CallSession>>,
) {
    for _ in 0..FINALIZE_SETTLE_PASSES {
        let _ = wait_for_capture_quiesce_with_timeout(state, guild_id, FINALIZE_SETTLE_TIMEOUT).await;

        let pending = flush_pending_buffers_for_export(state, guild_id).await;
        if !pending.is_empty() {
            let mut session = session_lock.write().await;
            upsert_utterances(&mut session.transcript, pending);
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
            .buffers
            .iter()
            .any(|e| e.key().0 == guild_id && !e.value().pcm.is_empty());

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

    while counter.load(Ordering::SeqCst) > 0 {
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

pub(super) async fn wait_for_capture_quiesce_with_timeout(
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

    while counter.load(Ordering::SeqCst) > 0 {
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

pub(super) fn upsert_utterances(transcript: &mut Vec<Utterance>, incoming: Vec<Utterance>) {
    for utterance in incoming {
        if let Some(existing) = transcript
            .iter_mut()
            .find(|u| u.revision_id == utterance.revision_id)
        {
            if existing.is_final && !utterance.is_final {
                continue;
            }
            *existing = utterance;
        } else {
            transcript.push(utterance);
        }
    }

    transcript.sort_by_key(|u| u.start_ts);
}

pub(super) async fn flush_pending_buffers_for_export(
    state: &Arc<AppState>,
    guild_id: GuildId,
) -> Vec<Utterance> {
    let user_keys: Vec<(GuildId, UserId)> = state
        .buffers
        .iter()
        .map(|e| *e.key())
        .filter(|(g, _)| *g == guild_id)
        .collect();
    let mut out = Vec::new();

    for user_key in user_keys {
        let user_id = user_key.1;
        let mut start_ts = None;
        let mut pcm = Vec::new();
        let mut revision_seq = None;

        if let Some(mut entry) = state.buffers.get_mut(&user_key) {
            if let Some(mut denoiser) = state.denoisers.get_mut(&user_key) {
                entry.pcm.extend(denoiser.flush_pending());
            }

            if entry.pcm.is_empty() {
                continue;
            }

            start_ts = Some(entry.utterance_start.take().unwrap_or_else(Instant::now));
            revision_seq = Some(entry.current_revision_seq.take().unwrap_or_else(|| {
                let seq = entry.next_revision_seq;
                entry.next_revision_seq = entry.next_revision_seq.wrapping_add(1);
                seq
            }));
            pcm = std::mem::take(&mut entry.pcm);
            entry.silent_ticks = 0;
            entry.last_preview_samples = 0;
            entry.last_preview_text = None;
            entry.frozen_prefix_words = 0;
            entry.stable_preview_streak = 0;
        }

        let Some(start_ts) = start_ts else {
            continue;
        };
        let Some(revision_seq) = revision_seq else {
            continue;
        };

        if let Some(text) = transcribe_finalized_mono_pcm(state, pcm).await {
            out.push(Utterance {
                user_id,
                start_ts,
                revision_id: make_revision_id(user_id, revision_seq),
                is_final: true,
                text,
            });
        }
    }

    for user_key in state
        .denoisers
        .iter()
        .map(|e| *e.key())
        .filter(|(g, _)| *g == guild_id)
        .collect::<Vec<_>>()
    {
        state.denoisers.remove(&user_key);
    }

    out
}

async fn transcribe_finalized_mono_pcm(state: &Arc<AppState>, pcm: Vec<f32>) -> Option<String> {
    if let Some(final_asr) = state.final_asr.as_ref() {
        if let Some(text) = transcribe_mono_pcm(Arc::clone(final_asr), pcm.clone()).await {
            return Some(text);
        }
    }

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
