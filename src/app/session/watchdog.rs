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
    SpeakingUpdateHandler, SsrcMap, Streams, VoiceTickHandler,
};

const WATCHDOG_CADENCE: Duration = Duration::from_secs(2);
const RECEIVE_EVIDENCE_WINDOW: Duration = Duration::from_secs(30);
const RECOVERY_VERIFICATION_TIMEOUT: Duration = Duration::from_secs(30);
const VOICE_OPERATION_TIMEOUT: Duration = Duration::from_secs(25);
const MAX_RECOVERY_ATTEMPTS: u32 = 3;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    ssrc_to_user: &SsrcMap,
    guild_id: GuildId,
    problems: &[ReceiveProblem],
) -> Vec<VerificationTarget> {
    problems
        .iter()
        .map(|problem| match problem {
            ReceiveProblem::Source(ssrc, _) => ssrc_to_user
                .get(&(guild_id, *ssrc))
                .map(|user| VerificationTarget::User(*user))
                .unwrap_or(VerificationTarget::Ssrc(*ssrc)),
            ReceiveProblem::Driver => VerificationTarget::AnyMappedMedia,
        })
        .collect()
}

fn audio_verified(
    ssrc_to_user: &SsrcMap,
    guild_id: GuildId,
    snapshot: &HealthSnapshot,
    verification: &Verification,
) -> bool {
    verification.targets.iter().all(|target| match target {
        VerificationTarget::User(user_id) => ssrc_to_user.iter().any(|entry| {
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

fn merge_verification_targets(
    existing: &mut Vec<VerificationTarget>,
    targets: impl IntoIterator<Item = VerificationTarget>,
) {
    for target in targets {
        if !existing.contains(&target) {
            existing.push(target);
        }
    }
}

fn retry_delay(attempts: u32) -> Duration {
    Duration::from_secs(10u64 << attempts)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecoveryDecision {
    Wait,
    Recover,
    Exhausted,
}

fn recovery_decision(
    observed_problems: &[ReceiveProblem],
    recovery_required: bool,
    attempts: u32,
    now: Instant,
    next_attempt_at: Instant,
) -> RecoveryDecision {
    if !recovery_required && observed_problems.is_empty() {
        RecoveryDecision::Wait
    } else if attempts >= MAX_RECOVERY_ATTEMPTS {
        RecoveryDecision::Exhausted
    } else if now >= next_attempt_at {
        RecoveryDecision::Recover
    } else {
        RecoveryDecision::Wait
    }
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
    let mut recovery_attempts = 0u32;
    let mut next_attempt_at = Instant::now();
    let mut verification: Option<Verification> = None;
    let mut recovery_targets = Vec::new();
    loop {
        tokio::time::sleep(WATCHDOG_CADENCE).await;
        if !is_current(&state, guild_id, &runtime) {
            return;
        }
        let now = Instant::now();
        let current = runtime.receive_health.snapshot();
        if !runtime.receive_health.driver_disconnected()
            && !runtime.recovery_needs_rejoin.load(Ordering::SeqCst)
            && verification.as_ref().is_some_and(|pending| {
                audio_verified(&state.ssrc_to_user, guild_id, &current, pending)
            })
        {
            tracing::info!(guild = %guild_id, attempts = recovery_attempts,
                "voice receive recovered: affected audio sources verified");
            verification = None;
            recovery_targets.clear();
            recovery_attempts = 0;
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
        let verification_expired = verification
            .as_ref()
            .is_some_and(|pending| pending.started.elapsed() >= RECOVERY_VERIFICATION_TIMEOUT);
        let recovery_required = needs_rejoin || verification_expired;
        let decision = recovery_decision(
            &persistent,
            recovery_required,
            recovery_attempts,
            now,
            next_attempt_at,
        );
        if decision == RecoveryDecision::Wait {
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
        if decision == RecoveryDecision::Exhausted {
            tracing::error!(guild = %guild_id, attempts = recovery_attempts,
                "voice receive recovery exhausted; ending transcription session");
            if let Err(error) = super::finalize::finalize_call_for_receive_failure(
                &ctx,
                &state,
                guild_id,
                &runtime,
                recovery_attempts,
            )
            .await
            {
                tracing::error!(guild = %guild_id,
                    "failed to finalize unrecoverable voice session: {error:#}");
            }
            return;
        }
        for problem in &persistent {
            tracing::warn!(guild = %guild_id, ?problem, "receive failure observed");
        }
        merge_verification_targets(
            &mut recovery_targets,
            verification_targets(&state.ssrc_to_user, guild_id, &persistent),
        );
        let targets = if recovery_targets.is_empty() {
            vec![VerificationTarget::AnyMappedMedia]
        } else {
            recovery_targets.clone()
        };
        recovery_attempts += 1;
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
                tracing::info!(guild = %guild_id, attempts = recovery_attempts,
                    "voice rejoined; awaiting mapped decoded packets from affected sources");
            }
            Ok(Ok(false)) => return,
            Ok(Err(error)) => {
                tracing::warn!(guild = %guild_id, attempts = recovery_attempts,
                    "voice receive recovery failed: {error:#}");
            }
            Err(_) => {
                tracing::warn!(guild = %guild_id, attempts = recovery_attempts,
                    "voice receive recovery attempt timed out");
            }
        }
        next_attempt_at = Instant::now() + retry_delay(recovery_attempts);
    }
}

async fn recover_receive(
    ctx: &Context,
    state: &Arc<AppState>,
    guild_id: GuildId,
    voice_channel: ChannelId,
    runtime: Arc<GuildRuntime>,
) -> anyhow::Result<bool> {
    let lifecycle_lock = state
        .session_start_locks
        .entry(guild_id)
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone();
    let _lifecycle_guard = lifecycle_lock.lock().await;
    if !is_current(state, guild_id, &runtime) {
        return Ok(false);
    }
    // Once recovery starts, keep retrying to a terminal outcome even if the
    // original packet evidence becomes stale while a remove/join attempt fails.
    runtime.recovery_needs_rejoin.store(true, Ordering::SeqCst);
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
    for buffered in detach_audio_buffers(&state.streams, guild_id) {
        let user_id = buffered.user_id;
        let mut pcm = buffered.pcm;
        trim_finalize_tail(&mut pcm, buffered.silent_ticks);
        if let Err(rejection) =
            should_dispatch_chunk(&pcm, buffered.voiced_ticks, buffered.noise_rms)
        {
            tracing::debug!(guild = %guild_id, user = %user_id, reason = rejection.reason,
                "recovery tail did not pass speech gate");
            continue;
        }
        let start_ts = buffered.utterance_start.unwrap_or_else(Instant::now);
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
                            user_id,
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
                Err(error) => tracing::warn!(guild = %guild_id, user = %user_id,
                    "recovery tail transcription failed: {error:#}"),
            }
            runtime
                .transcription_inflight
                .fetch_sub(1, Ordering::SeqCst);
        });
    }
}

struct DetachedAudioBuffer {
    user_id: UserId,
    pcm: Vec<f32>,
    silent_ticks: u32,
    voiced_ticks: u32,
    utterance_start: Option<Instant>,
    noise_rms: f32,
}

fn detach_audio_buffers(streams: &Streams, guild_id: GuildId) -> Vec<DetachedAudioBuffer> {
    let keys: Vec<(GuildId, UserId)> = streams
        .iter()
        .map(|e| *e.key())
        .filter(|(g, _)| *g == guild_id)
        .collect();
    let mut detached = Vec::with_capacity(keys.len());
    for key in keys {
        let Some((_key, mut stream)) = streams.remove(&key) else {
            continue;
        };
        let pcm = std::mem::take(&mut stream.buffer.pcm);
        if pcm.is_empty() {
            continue;
        }
        detached.push(DetachedAudioBuffer {
            user_id: key.1,
            pcm,
            silent_ticks: stream.buffer.silent_ticks,
            voiced_ticks: stream.buffer.voiced_ticks,
            utterance_start: stream.buffer.utterance_start,
            noise_rms: stream.frontend.noise_rms_ema(),
        });
    }
    detached
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::healthcheck::ReceiveHealth;

    #[test]
    fn quiet_session_waits_but_expired_verification_recovers() {
        let now = Instant::now();
        assert_eq!(
            recovery_decision(&[], false, 0, now, now),
            RecoveryDecision::Wait
        );
        assert_eq!(
            recovery_decision(&[], true, 0, now, now),
            RecoveryDecision::Recover
        );
    }

    #[test]
    fn repeated_failures_back_off_and_require_fresh_evidence() {
        let now = Instant::now();
        let issue = [ReceiveProblem::Source(7, SourceProblem::Decode)];
        assert_eq!(retry_delay(1), Duration::from_secs(20));
        assert_eq!(retry_delay(MAX_RECOVERY_ATTEMPTS), Duration::from_secs(80));
        assert_eq!(
            recovery_decision(&issue, false, 1, now, now + retry_delay(1)),
            RecoveryDecision::Wait
        );
        assert_eq!(
            recovery_decision(&issue, false, 1, now, now),
            RecoveryDecision::Recover
        );
        assert_eq!(
            recovery_decision(&issue, false, MAX_RECOVERY_ATTEMPTS, now, now),
            RecoveryDecision::Exhausted
        );
        assert_eq!(
            recovery_decision(&[], true, MAX_RECOVERY_ATTEMPTS, now, now),
            RecoveryDecision::Exhausted
        );
        assert_eq!(
            recovery_decision(&[], false, MAX_RECOVERY_ATTEMPTS, now, now),
            RecoveryDecision::Wait
        );
    }

    #[test]
    fn verification_follows_users_across_ssrc_changes_and_requires_every_user() {
        let guild_id = GuildId::new(1);
        let affected_user = UserId::new(10);
        let other_user = UserId::new(20);
        let mappings = SsrcMap::new();
        mappings.insert((guild_id, 100), affected_user);
        mappings.insert((guild_id, 200), other_user);
        let health = ReceiveHealth::default();
        for _ in 0..3 {
            health.packet(200);
            health.decoded(200, true, true);
        }
        let verification = Verification {
            targets: vec![VerificationTarget::User(affected_user)],
            started: Instant::now(),
            reported_unverified: false,
        };

        assert!(!audio_verified(
            &mappings,
            guild_id,
            &health.snapshot(),
            &verification
        ));

        mappings.remove(&(guild_id, 100));
        mappings.insert((guild_id, 101), affected_user);
        for _ in 0..3 {
            health.packet(101);
            health.decoded(101, true, true);
        }
        assert!(audio_verified(
            &mappings,
            guild_id,
            &health.snapshot(),
            &verification
        ));

        let both_users = Verification {
            targets: vec![
                VerificationTarget::User(affected_user),
                VerificationTarget::User(UserId::new(30)),
            ],
            started: Instant::now(),
            reported_unverified: false,
        };
        assert!(!audio_verified(
            &mappings,
            guild_id,
            &health.snapshot(),
            &both_users
        ));
    }

    #[test]
    fn recovery_targets_map_known_users_and_deduplicate_across_attempts() {
        let guild_id = GuildId::new(1);
        let user_id = UserId::new(10);
        let mappings = SsrcMap::new();
        mappings.insert((guild_id, 100), user_id);
        let problems = [
            ReceiveProblem::Source(100, SourceProblem::Decode),
            ReceiveProblem::Source(200, SourceProblem::Mapping),
            ReceiveProblem::Driver,
        ];

        let targets = verification_targets(&mappings, guild_id, &problems);
        assert_eq!(
            targets,
            vec![
                VerificationTarget::User(user_id),
                VerificationTarget::Ssrc(200),
                VerificationTarget::AnyMappedMedia,
            ]
        );

        let mut accumulated = vec![VerificationTarget::User(user_id)];
        merge_verification_targets(&mut accumulated, targets);
        assert_eq!(
            accumulated,
            vec![
                VerificationTarget::User(user_id),
                VerificationTarget::Ssrc(200),
                VerificationTarget::AnyMappedMedia,
            ]
        );
    }

    #[test]
    fn unknown_source_verification_requires_that_specific_ssrc() {
        let guild_id = GuildId::new(1);
        let mappings = SsrcMap::new();
        let verification = Verification {
            targets: vec![
                VerificationTarget::Ssrc(200),
                VerificationTarget::AnyMappedMedia,
            ],
            started: Instant::now(),
            reported_unverified: false,
        };
        let health = ReceiveHealth::default();
        for _ in 0..3 {
            health.packet(201);
            health.decoded(201, true, true);
        }
        assert!(!audio_verified(
            &mappings,
            guild_id,
            &health.snapshot(),
            &verification
        ));

        for _ in 0..3 {
            health.packet(200);
            health.decoded(200, true, true);
        }
        assert!(audio_verified(
            &mappings,
            guild_id,
            &health.snapshot(),
            &verification
        ));
    }

    #[test]
    fn detaching_recovery_audio_preserves_pcm_and_other_guilds() {
        let guild_id = GuildId::new(1);
        let other_guild = GuildId::new(2);
        let user_id = UserId::new(10);
        let started = Instant::now();
        let streams = Streams::new();
        streams.insert((guild_id, user_id), Default::default());
        {
            let mut stream = streams
                .get_mut(&(guild_id, user_id))
                .expect("target stream exists");
            stream.buffer.pcm = vec![0.1, 0.2, 0.3];
            stream.buffer.silent_ticks = 2;
            stream.buffer.voiced_ticks = 7;
            stream.buffer.utterance_start = Some(started);
        }
        streams.insert((guild_id, UserId::new(11)), Default::default());
        streams.insert((other_guild, UserId::new(20)), Default::default());

        let detached = detach_audio_buffers(&streams, guild_id);

        assert_eq!(detached.len(), 1);
        assert_eq!(detached[0].user_id, user_id);
        assert_eq!(detached[0].pcm, vec![0.1, 0.2, 0.3]);
        assert_eq!(detached[0].silent_ticks, 2);
        assert_eq!(detached[0].voiced_ticks, 7);
        assert_eq!(detached[0].utterance_start, Some(started));
        assert!(!streams.contains_key(&(guild_id, user_id)));
        assert!(!streams.contains_key(&(guild_id, UserId::new(11))));
        assert!(streams.contains_key(&(other_guild, UserId::new(20))));
    }
}
