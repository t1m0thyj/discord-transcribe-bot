use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serenity::all::{ChannelId, GuildId, UserId};
use serenity::prelude::Context;
use songbird::events::{CoreEvent, Event};

use super::super::healthcheck::{assess, mapped_audio_observed, HealthSnapshot, SourceProblem};
use super::super::{AppState, GuildRuntime, Utterance};
use crate::asr::{
    clear_unknown_ssrc_audio_for_guild, should_dispatch_chunk, transcribe_mono_pcm,
    trim_finalize_tail, ClientDisconnectHandler, DriverHealthHandler, RtpPacketHandler,
    SpeakingUpdateHandler, VoiceTickHandler,
};

const WATCHDOG_CADENCE: Duration = Duration::from_secs(2);
const RECEIVE_EVIDENCE_WINDOW: Duration = Duration::from_secs(30);
const VOICE_OPERATION_TIMEOUT: Duration = Duration::from_secs(25);

pub struct VoiceHandlerAttachContext {
    pub http: Arc<serenity::http::Http>,
    pub guild_id: GuildId,
    pub text_channel: ChannelId,
    pub voice_channel: ChannelId,
    pub call_lock: Arc<tokio::sync::Mutex<songbird::Call>>,
    pub runtime: Arc<GuildRuntime>,
}

pub async fn attach_voice_mapping_handlers(
    state: &Arc<AppState>,
    guild_id: GuildId,
    call_lock: &Arc<tokio::sync::Mutex<songbird::Call>>,
    runtime: &Arc<GuildRuntime>,
) {
    let mut call = call_lock.lock().await;
    let generation = runtime.receive_generation.load(Ordering::SeqCst);
    call.add_global_event(
        Event::Core(CoreEvent::SpeakingStateUpdate),
        SpeakingUpdateHandler {
            guild_id,
            ssrc_to_user: Arc::clone(&state.ssrc_to_user),
            health: Arc::clone(&runtime.receive_health),
            generation: Arc::clone(&runtime.receive_generation),
            expected_generation: generation,
        },
    );
    call.add_global_event(
        Event::Core(CoreEvent::ClientDisconnect),
        ClientDisconnectHandler {
            guild_id,
            ssrc_to_user: Arc::clone(&state.ssrc_to_user),
            health: Arc::clone(&runtime.receive_health),
            generation: Arc::clone(&runtime.receive_generation),
            expected_generation: generation,
        },
    );
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
    let generation = runtime.receive_generation.load(Ordering::SeqCst);
    let mut call = call_lock.lock().await;
    for event in [
        CoreEvent::DriverDisconnect,
        CoreEvent::DriverReconnect,
        CoreEvent::DriverConnect,
    ] {
        call.add_global_event(
            Event::Core(event),
            DriverHealthHandler {
                health: Arc::clone(&runtime.receive_health),
                generation: Arc::clone(&runtime.receive_generation),
                expected_generation: generation,
            },
        );
    }
    call.add_global_event(
        Event::Core(CoreEvent::RtpPacket),
        RtpPacketHandler {
            runtime: Arc::clone(&runtime),
            generation,
        },
    );
    call.add_global_event(
        Event::Core(CoreEvent::VoiceTick),
        VoiceTickHandler {
            http,
            text_channel,
            runtime: Arc::clone(&runtime),
            generation,
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

fn is_current(state: &AppState, guild_id: GuildId, runtime: &Arc<GuildRuntime>) -> bool {
    state
        .guild_runtimes
        .get(&guild_id)
        .is_some_and(|current| Arc::ptr_eq(current.value(), runtime))
        && state.active_calls.contains_key(&guild_id)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReceiveProblem {
    Source(u32, SourceProblem),
    Driver,
}

#[derive(Clone, Copy)]
enum VerificationTarget {
    User(UserId),
    Ssrc(u32),
    AnyMappedMedia,
}

struct Verification {
    targets: Vec<VerificationTarget>,
    started: Instant,
    reported_unverified: bool,
}

fn verification_targets(
    state: &AppState,
    guild_id: GuildId,
    problems: &[ReceiveProblem],
) -> Vec<VerificationTarget> {
    problems
        .iter()
        .map(|problem| match problem {
            ReceiveProblem::Source(ssrc, _) => state
                .ssrc_to_user
                .get(&(guild_id, *ssrc))
                .map(|user| VerificationTarget::User(*user))
                .unwrap_or(VerificationTarget::Ssrc(*ssrc)),
            ReceiveProblem::Driver => VerificationTarget::AnyMappedMedia,
        })
        .collect()
}

fn audio_verified(
    state: &AppState,
    guild_id: GuildId,
    snapshot: &HealthSnapshot,
    verification: &Verification,
) -> bool {
    verification.targets.iter().all(|target| match target {
        VerificationTarget::User(user_id) => state.ssrc_to_user.iter().any(|entry| {
            entry.key().0 == guild_id
                && *entry.value() == *user_id
                && mapped_audio_observed(snapshot, entry.key().1, 3)
        }),
        VerificationTarget::Ssrc(ssrc) => mapped_audio_observed(snapshot, *ssrc, 3),
        VerificationTarget::AnyMappedMedia => {
            snapshot.0.values().any(|counts| counts.mapped_packets >= 3)
        }
    })
}

fn retry_delay(attempts: u32) -> Duration {
    Duration::from_secs(10u64.saturating_mul(1 << attempts.min(5)))
}

fn recovery_due(
    observed_problems: &[ReceiveProblem],
    known_disconnected: bool,
    now: Instant,
    next_attempt_at: Instant,
) -> bool {
    (known_disconnected || !observed_problems.is_empty()) && now >= next_attempt_at
}

/// Silence is idle. A fresh speaking indication without media or a failed
/// driver gives independent evidence when packet callbacks cannot fire.
pub async fn receive_watchdog(
    ctx: Context,
    state: Arc<AppState>,
    guild_id: GuildId,
    voice_channel: ChannelId,
    runtime: Arc<GuildRuntime>,
) {
    let mut history = VecDeque::from([(Instant::now(), runtime.receive_health.snapshot())]);
    let mut previous_problems = Vec::new();
    let mut failures = 0u32;
    let mut next_attempt_at = Instant::now();
    let mut verification: Option<Verification> = None;
    loop {
        tokio::time::sleep(WATCHDOG_CADENCE).await;
        if !is_current(&state, guild_id, &runtime) {
            return;
        }
        let now = Instant::now();
        let current = runtime.receive_health.snapshot();
        if !runtime.receive_health.driver_disconnected()
            && !runtime.recovery_needs_rejoin.load(Ordering::SeqCst)
            && verification
                .as_ref()
                .is_some_and(|pending| audio_verified(&state, guild_id, &current, pending))
        {
            tracing::info!(guild = %guild_id, attempts = failures,
                "voice receive recovered: affected audio sources verified");
            verification = None;
            failures = 0;
            next_attempt_at = now;
            runtime
                .receive_verification_pending
                .store(false, Ordering::SeqCst);
        }
        if let Some(pending) = &mut verification {
            if !pending.reported_unverified && pending.started.elapsed() >= Duration::from_secs(15)
            {
                tracing::warn!(guild = %guild_id,
                    "voice rejoin completed but affected audio remains unverified; waiting for fresh media evidence");
                pending.reported_unverified = true;
            }
        }

        history.push_back((now, current.clone()));
        while history.len() > 1
            && now.saturating_duration_since(history[1].0) >= RECEIVE_EVIDENCE_WINDOW
        {
            history.pop_front();
        }
        let baseline = &history.front().expect("health history has a snapshot").1;
        let problems: Vec<_> = assess(baseline, &current, now)
            .into_iter()
            .map(|(ssrc, problem)| ReceiveProblem::Source(ssrc, problem))
            .collect();
        let mut persistent: Vec<_> = problems
            .iter()
            .copied()
            .filter(|problem| previous_problems.contains(problem))
            .collect();
        previous_problems = problems;
        if runtime.receive_health.driver_disconnected() {
            persistent.push(ReceiveProblem::Driver);
        }
        let needs_rejoin = runtime.recovery_needs_rejoin.load(Ordering::SeqCst);
        if !recovery_due(&persistent, needs_rejoin, now, next_attempt_at) {
            continue;
        }
        let bot_id = ctx.cache.current_user().id;
        let non_bot_present = ctx.cache.guild(guild_id).is_some_and(|guild| {
            guild.voice_states.iter().any(|(user_id, voice)| {
                *user_id != bot_id && voice.channel_id == Some(voice_channel)
            })
        });
        if !non_bot_present {
            continue;
        }
        for problem in &persistent {
            tracing::warn!(guild = %guild_id, ?problem, "receive failure observed");
        }
        let targets = if persistent.is_empty() {
            verification
                .as_ref()
                .map(|pending| pending.targets.clone())
                .unwrap_or_else(|| vec![VerificationTarget::AnyMappedMedia])
        } else {
            verification_targets(&state, guild_id, &persistent)
        };
        failures = failures.saturating_add(1);
        match tokio::time::timeout(
            Duration::from_secs(60),
            recover_receive(&ctx, &state, guild_id, voice_channel, Arc::clone(&runtime)),
        )
        .await
        {
            Ok(Ok(true)) => {
                runtime
                    .receive_verification_pending
                    .store(true, Ordering::SeqCst);
                verification = Some(Verification {
                    targets,
                    started: Instant::now(),
                    reported_unverified: false,
                });
                history.clear();
                history.push_back((Instant::now(), runtime.receive_health.snapshot()));
                previous_problems.clear();
                tracing::info!(guild = %guild_id, attempts = failures,
                    "voice rejoined; awaiting mapped decoded packets from affected sources");
            }
            Ok(Ok(false)) => return,
            Ok(Err(error)) => {
                tracing::warn!(guild = %guild_id, failures, "voice receive recovery failed: {error:#}");
            }
            Err(_) => {
                tracing::warn!(guild = %guild_id, failures, "voice receive recovery attempt timed out");
            }
        }
        next_attempt_at = Instant::now() + retry_delay(failures);
    }
}

async fn recover_receive(
    ctx: &Context,
    state: &Arc<AppState>,
    guild_id: GuildId,
    voice_channel: ChannelId,
    runtime: Arc<GuildRuntime>,
) -> anyhow::Result<bool> {
    let _recovery_guard = runtime.recovery_lock.lock().await;
    let lifecycle_lock = state
        .session_start_locks
        .entry(guild_id)
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone();
    let _lifecycle_guard = lifecycle_lock.lock().await;
    if !is_current(state, guild_id, &runtime) {
        return Ok(false);
    }
    let manager = songbird::get(ctx)
        .await
        .ok_or_else(|| anyhow::anyhow!("songbird voice manager unavailable"))?
        .clone();
    if let Err(error) = tokio::time::timeout(VOICE_OPERATION_TIMEOUT, manager.remove(guild_id))
        .await
        .map_err(|_| anyhow::anyhow!("voice remove timed out"))?
    {
        if manager.get(guild_id).is_some() {
            return Err(error.into());
        }
    }

    runtime.recovery_needs_rejoin.store(true, Ordering::SeqCst);
    runtime.receive_generation.fetch_add(1, Ordering::SeqCst);

    // The old Songbird call has stopped writing; preserve tails independently of reconnect.
    detach_and_transcribe_buffers(state, guild_id, &runtime);
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
    runtime.receive_health.clear();

    let call_lock = manager.get_or_insert(guild_id);
    attach_voice_mapping_handlers(state, guild_id, &call_lock, &runtime).await;
    let call_lock = tokio::time::timeout(
        VOICE_OPERATION_TIMEOUT,
        manager.join(guild_id, voice_channel),
    )
    .await
    .map_err(|_| anyhow::anyhow!("voice join timed out"))??;
    let session_lock = state
        .active_calls
        .get(&guild_id)
        .ok_or_else(|| anyhow::anyhow!("active call disappeared during recovery"))?
        .clone();
    let text_channel = session_lock.read().await.text_channel;
    attach_voice_handlers(
        state,
        VoiceHandlerAttachContext {
            http: Arc::clone(&ctx.http),
            guild_id,
            text_channel,
            voice_channel,
            call_lock,
            runtime: Arc::clone(&runtime),
        },
    )
    .await;
    let current = state
        .guild_runtimes
        .get(&guild_id)
        .is_some_and(|entry| Arc::ptr_eq(entry.value(), &runtime));
    if current {
        runtime.recovery_needs_rejoin.store(false, Ordering::SeqCst);
    }
    Ok(true)
}

pub(super) fn detach_and_transcribe_buffers(
    state: &Arc<AppState>,
    guild_id: GuildId,
    runtime: &Arc<GuildRuntime>,
) {
    let keys: Vec<(GuildId, UserId)> = state
        .streams
        .iter()
        .map(|e| *e.key())
        .filter(|(g, _)| *g == guild_id)
        .collect();
    for key in keys {
        let Some((_key, mut stream)) = state.streams.remove(&key) else {
            continue;
        };
        let mut pcm = std::mem::take(&mut stream.buffer.pcm);
        if pcm.is_empty() {
            continue;
        }
        trim_finalize_tail(&mut pcm, stream.buffer.silent_ticks);
        if let Err(rejection) = should_dispatch_chunk(
            &pcm,
            stream.buffer.voiced_ticks,
            stream.frontend.noise_rms_ema(),
        ) {
            runtime.dispatch_gate_total.fetch_add(1, Ordering::SeqCst);
            tracing::debug!(guild = %guild_id, user = %key.1, reason = rejection.reason,
                "recovery tail did not pass speech gate");
            continue;
        }
        let start_ts = stream.buffer.utterance_start.unwrap_or_else(Instant::now);
        let runtime = Arc::clone(runtime);
        let asr = Arc::clone(&state.asr);
        runtime
            .transcription_inflight
            .fetch_add(1, Ordering::SeqCst);
        tokio::spawn(async move {
            match transcribe_mono_pcm(asr, pcm).await {
                Ok(Some(text)) => {
                    runtime
                        .transcript_pending_commits
                        .fetch_add(1, Ordering::SeqCst);
                    if runtime
                        .utterance_tx
                        .send(Utterance {
                            user_id: key.1,
                            start_ts,
                            text,
                        })
                        .await
                        .is_err()
                    {
                        runtime
                            .transcript_pending_commits
                            .fetch_sub(1, Ordering::SeqCst);
                    }
                }
                Ok(None) => {}
                Err(error) => tracing::warn!(guild = %guild_id, user = %key.1,
                    "recovery tail transcription failed: {error:#}"),
            }
            runtime
                .transcription_inflight
                .fetch_sub(1, Ordering::SeqCst);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unverified_quiet_rejoin_does_not_loop() {
        let now = Instant::now();
        assert!(!recovery_due(&[], false, now, now));
        assert!(recovery_due(&[], true, now, now));
    }

    #[test]
    fn repeated_failures_back_off_and_require_fresh_evidence() {
        let now = Instant::now();
        let issue = [ReceiveProblem::Source(7, SourceProblem::Decode)];
        assert_eq!(retry_delay(1), Duration::from_secs(20));
        assert_eq!(retry_delay(5), Duration::from_secs(320));
        assert!(!recovery_due(&issue, false, now, now + retry_delay(1)));
        assert!(recovery_due(&issue, false, now, now));
        assert!(!recovery_due(&[], false, now, now));
    }
}
