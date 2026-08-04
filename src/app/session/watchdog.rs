use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use serenity::all::{ChannelId, GuildId, UserId};
use serenity::prelude::Context;
use songbird::events::{CoreEvent, Event};

use super::super::{
    AppState, GuildRuntime, STARTUP_RECEIVE_RECOVERY_MAX_ATTEMPTS, STARTUP_RECEIVE_WATCHDOG_DELAY,
    STEADY_STATE_NO_PROGRESS_TIMEOUT, STEADY_STATE_WATCHDOG_CADENCE,
};
use crate::audio::{
    clear_unknown_ssrc_audio_for_guild, ClientDisconnectHandler, SpeakingUpdateHandler,
    VoiceTickHandler,
};

pub struct VoiceHandlerAttachContext {
    pub http: Arc<serenity::http::Http>,
    pub guild_id: GuildId,
    pub text_channel: ChannelId,
    pub voice_channel: ChannelId,
    pub call_lock: Arc<tokio::sync::Mutex<songbird::Call>>,
    pub runtime: Arc<GuildRuntime>,
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
        let _ = super::finalize::finalize_call_for_guild(&ctx, &state, guild_id).await;
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
