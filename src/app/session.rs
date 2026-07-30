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
use tokio::sync::{mpsc, oneshot};

use super::{
    AppState, STARTUP_RECEIVE_RECOVERY_MAX_ATTEMPTS,
    STARTUP_RECEIVE_WATCHDOG_DELAY, STEADY_STATE_NO_PROGRESS_TIMEOUT,
    STEADY_STATE_WATCHDOG_CADENCE, ThreadContext, Utterance,
    UtteranceStage,
};
use crate::audio::{ClientDisconnectHandler, SpeakingUpdateHandler, VoiceTickHandler};
use crate::transcription::StreamingDecoderCommand;

pub struct VoiceHandlerAttachContext {
    pub http: Arc<serenity::http::Http>,
    pub guild_id: GuildId,
    pub text_channel: ChannelId,
    pub voice_channel: ChannelId,
    pub call_lock: Arc<tokio::sync::Mutex<songbird::Call>>,
    pub streaming_decoder_tx: mpsc::Sender<StreamingDecoderCommand>,
    pub decode_activity: Arc<AtomicUsize>,
    pub chunks_accepted_activity: Arc<AtomicUsize>,
    pub decode_failure_activity: Arc<AtomicUsize>,
    pub decoder_queue_dropped: Arc<AtomicUsize>,
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

    if let Some(decoder_tx) = state
        .streaming_decoder_senders
        .get(&guild_id)
        .map(|v| v.value().clone())
    {
        let (ack_tx, ack_rx) = oneshot::channel();
        let _ = decoder_tx
            .send(crate::transcription::StreamingDecoderCommand::FlushAll {
                respond_to: ack_tx,
                observed_at: Instant::now(),
            })
            .await;
        let _ = tokio::time::timeout(Duration::from_secs(5), ack_rx).await;
    }
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

    state.utterance_senders.remove(&guild_id);
    state.streaming_decoder_senders.remove(&guild_id);
    for key in state
        .live_partial_text
        .iter()
        .map(|e| *e.key())
        .filter(|(g, _)| *g == guild_id)
        .collect::<Vec<_>>()
    {
        state.live_partial_text.remove(&key);
    }

    let session = session_lock.read().await;
    let mut transcript = load_persisted_transcript(
        &session.transcript_jsonl_path,
        session.started_mono,
    );
    if transcript.is_empty() {
        transcript = session.transcript.clone();
    } else {
        upsert_utterances(&mut transcript, session.transcript.clone());
    }

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
    state.decoded_audio_activity.remove(&guild_id);
    state.chunks_accepted_activity.remove(&guild_id);
    state.decode_failure_activity.remove(&guild_id);
    state.decoder_queue_dropped.remove(&guild_id);
    state.unmapped_ssrc_activity.remove(&guild_id);
    state.decoder_thread_alive.remove(&guild_id);
    state.offline_finalize_worker_alive.remove(&guild_id);
    state.offline_finalize_rtf_milli_ewma.remove(&guild_id);
    state.offline_finalize_empty.remove(&guild_id);
    state.offline_finalize_dropped.remove(&guild_id);
    state.refinement_rejected.remove(&guild_id);
    state.transcription_started_notified.remove(&guild_id);

    Ok(())
}

#[derive(Deserialize)]
struct PersistedUtterance {
    revision_id: u64,
    user_id: u64,
    start_offset_ms: u64,
    #[serde(default)]
    stage: Option<UtteranceStage>,
    is_final: bool,
    text: String,
    #[serde(default)]
    tokens: Vec<String>,
    #[serde(default)]
    token_timestamps_s: Vec<f32>,
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
        upsert_utterances(&mut out, vec![Utterance {
            user_id: UserId::new(item.user_id),
            start_ts: started_mono + Duration::from_millis(item.start_offset_ms),
            start_offset_ms: item.start_offset_ms,
            revision_id: item.revision_id,
            stage: item.stage.unwrap_or(if item.is_final {
                UtteranceStage::OfflineFinal
            } else {
                UtteranceStage::Partial
            }),
            is_final: item.is_final,
            text: item.text,
            tokens: item.tokens,
            token_timestamps_s: item.token_timestamps_s,
        }]);
    }

    out
}

pub async fn attach_voice_handlers(state: &Arc<AppState>, ctx: VoiceHandlerAttachContext) {
    let VoiceHandlerAttachContext {
        http,
        guild_id,
        text_channel,
        voice_channel,
        call_lock,
        streaming_decoder_tx,
        decode_activity,
        chunks_accepted_activity,
        decode_failure_activity,
        decoder_queue_dropped,
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
            streams: Arc::clone(&state.streams),
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
            streaming_decoder_tx,
            enable_denoiser: state.enable_denoiser,
            decode_activity,
            chunks_accepted_activity,
            decode_failure_activity,
            decoder_queue_dropped,
            unmapped_ssrc_activity,
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

    let Some(streaming_decoder_tx) = state
        .streaming_decoder_senders
        .get(&guild_id)
        .map(|v| v.value().clone())
    else {
        tracing::warn!(
            guild = %guild_id,
            attempt,
            "startup recovery retrying: missing streaming decoder sender state"
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
    let Some(chunks_accepted_activity) = state
        .chunks_accepted_activity
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        tracing::warn!(
            guild = %guild_id,
            attempt,
            "startup recovery retrying: missing accepted-chunks counter state"
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
    let Some(decoder_queue_dropped) = state
        .decoder_queue_dropped
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()))
    else {
        tracing::warn!(
            guild = %guild_id,
            attempt,
            "startup recovery retrying: missing decoder queue-drop counter state"
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
    chunks_accepted_activity.store(0, Ordering::SeqCst);
    decode_failure_activity.store(0, Ordering::SeqCst);
    decoder_queue_dropped.store(0, Ordering::SeqCst);
    unmapped_ssrc_activity.store(0, Ordering::SeqCst);

    attach_voice_handlers(
        &state,
        VoiceHandlerAttachContext {
            http: Arc::clone(&ctx.http),
            guild_id,
            text_channel,
            voice_channel,
            call_lock: Arc::clone(&call_lock),
            streaming_decoder_tx,
            decode_activity,
            chunks_accepted_activity,
            decode_failure_activity,
            decoder_queue_dropped,
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
    let mut last_chunks_accepted = 0usize;
    let mut last_progress = Instant::now();
    let mut decode_without_accept_since: Option<Instant> = None;
    let mut offline_finalize_warned = false;

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
            decode_without_accept_since = None;
            continue;
        }

        let decode_activity = state
            .decoded_audio_activity
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(0);
        let chunks_accepted = state
            .chunks_accepted_activity
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(0);
        let decoder_alive = state
            .decoder_thread_alive
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(false);
        let offline_finalize_alive = state
            .offline_finalize_worker_alive
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(false);

        if !decoder_alive {
            tracing::error!(
                guild = %guild_id,
                decode_activity,
                chunks_accepted,
                "steady-state watchdog detected dead decoder thread"
            );
            let _ = text_channel
                .say(
                    &ctx.http,
                    "Decoder thread exited unexpectedly; ending this session so you can /join again cleanly.",
                )
                .await;
            let _ = finalize_call_for_guild(&ctx, &state, guild_id).await;
            return;
        }

        if !offline_finalize_alive {
            if !offline_finalize_warned {
                tracing::warn!(
                    guild = %guild_id,
                    "steady-state watchdog detected dead offline finalize worker"
                );
                let _ = text_channel
                    .say(
                        &ctx.http,
                        "Offline refinement worker exited unexpectedly; continuing with stream-final transcripts for this session.",
                    )
                    .await;
                offline_finalize_warned = true;
            }
        } else {
            offline_finalize_warned = false;
        }

        if decode_activity > last_decode_activity && chunks_accepted == last_chunks_accepted {
            let since = decode_without_accept_since.get_or_insert_with(Instant::now);
            if since.elapsed() >= STEADY_STATE_NO_PROGRESS_TIMEOUT {
                tracing::error!(
                    guild = %guild_id,
                    decode_activity,
                    chunks_accepted,
                    "steady-state watchdog detected decode activity without accepted decoder chunks"
                );
                let _ = text_channel
                    .say(
                        &ctx.http,
                        "Decoded audio is arriving but decoder queue accepts no chunks; reinitializing voice receive.",
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
                last_chunks_accepted = state
                    .chunks_accepted_activity
                    .get(&guild_id)
                    .map(|v| v.load(Ordering::SeqCst))
                    .unwrap_or(0);
                decode_without_accept_since = None;
            } else {
                last_decode_activity = decode_activity;
            }
            continue;
        }

        decode_without_accept_since = None;

        if decode_activity > last_decode_activity || chunks_accepted > last_chunks_accepted {
            last_decode_activity = decode_activity;
            last_chunks_accepted = chunks_accepted;
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
        let decoder_queue_dropped = state
            .decoder_queue_dropped
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(0);
        let unmapped_ssrc = state
            .unmapped_ssrc_activity
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(0);
        let offline_finalize_dropped = state
            .offline_finalize_dropped
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(0);

        tracing::warn!(
            guild = %guild_id,
            decode_activity,
            chunks_accepted,
            decode_failures,
            decoder_queue_dropped,
            unmapped_ssrc,
            offline_finalize_dropped,
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
        last_chunks_accepted = state
            .chunks_accepted_activity
            .get(&guild_id)
            .map(|v| v.load(Ordering::SeqCst))
            .unwrap_or(0);
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

pub(super) fn upsert_utterances(transcript: &mut Vec<Utterance>, incoming: Vec<Utterance>) {
    for utterance in incoming {
        if let Some(existing) = transcript
            .iter_mut()
            .find(|u| u.revision_id == utterance.revision_id)
        {
            if existing.stage.precedence() > utterance.stage.precedence() {
                continue;
            }
            *existing = utterance;
        } else {
            transcript.push(utterance);
        }
    }

    transcript.sort_by_key(|u| u.start_ts);
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

        let total = utt.start_offset_ms / 1000;
        let hh = total / 3600;
        let mm = (total % 3600) / 60;
        let ss = total % 60;
        let stage_hint = match utt.stage {
            UtteranceStage::StreamFinal => " [stream-final]",
            _ => "",
        };
        lines.push(format!(
            "[{hh:02}:{mm:02}:{ss:02}] {display}{stage_hint}: {}",
            utt.text
        ));
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
