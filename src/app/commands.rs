use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::Context as _;
use serenity::all::{ChannelType, CommandDataOptionValue, CommandInteraction};
use serenity::prelude::Context;

use super::{
    AppState, LOG_DEFAULT_UTTERANCES, LOG_MAX_DISCORD_CHARS,
};
use super::autojoin::{normalized_autojoin_suffix, strip_known_autojoin_suffix};
use crate::asr::{decode_queue_capacity, decode_queue_depth};

pub(super) async fn handle_join(
    ctx: &Context,
    state: &Arc<AppState>,
    command: &CommandInteraction,
) -> anyhow::Result<String> {
    let guild_id = command.guild_id.context("join used outside guild")?;

    if state.active_calls.contains_key(&guild_id) {
        return Ok("Already connected and transcribing in this guild. Use /leave first if you want a fresh session.".to_string());
    }

    let channel_id = {
        let guild = ctx.cache.guild(guild_id).context("guild not in cache")?;
        guild
            .voice_states
            .get(&command.user.id)
            .and_then(|vs| vs.channel_id)
            .context("you are not connected to a voice channel")?
    };

    super::session::start_call_session(ctx, state, guild_id, channel_id, command.channel_id).await?;

    Ok(format!(
        "Joined <#{}>. Listening and waiting for speech...",
        channel_id.get()
    ))
}

pub(super) async fn handle_status(
    ctx: &Context,
    state: &Arc<AppState>,
    command: &CommandInteraction,
) -> anyhow::Result<String> {
    let guild_id = command.guild_id.context("status used outside guild")?;

    let Some(session_lock) = state.active_calls.get(&guild_id).map(|s| s.clone()) else {
        return Ok("No active transcription session. Use /join to start.".to_string());
    };

    let session = session_lock.read().await;
    let voice_channel = session.voice_channel;
    let utterance_count = session.transcript.len();
    let elapsed = session.started_mono.elapsed();
    drop(session);

    let runtime = state
        .guild_runtimes
        .get(&guild_id)
        .map(|v| Arc::clone(v.value()));

    let decoded_frames = runtime
        .as_ref()
        .map(|v| v.decoded_audio_activity.load(Ordering::SeqCst))
        .unwrap_or(0);
    let inflight = runtime
        .as_ref()
        .map(|v| v.transcription_inflight.load(Ordering::SeqCst))
        .unwrap_or(0);
    let pending_commits = runtime
        .as_ref()
        .map(|v| v.transcript_pending_commits.load(Ordering::SeqCst))
        .unwrap_or(0);
    let decode_failures = runtime
        .as_ref()
        .map(|v| v.decode_failure_activity.load(Ordering::SeqCst))
        .unwrap_or(0);
    let decode_jobs_total = runtime
        .as_ref()
        .map(|v| v.decode_jobs_total.load(Ordering::SeqCst))
        .unwrap_or(0);
    let decode_jobs_with_text = runtime
        .as_ref()
        .map(|v| v.decode_jobs_with_text.load(Ordering::SeqCst))
        .unwrap_or(0);
    let decode_audio_total_ms = runtime
        .as_ref()
        .map(|v| v.decode_audio_total_ms.load(Ordering::SeqCst))
        .unwrap_or(0);
    let decode_total_ms = runtime
        .as_ref()
        .map(|v| v.decode_total_ms.load(Ordering::SeqCst))
        .unwrap_or(0);
    let decode_queue_wait_total_ms = runtime
        .as_ref()
        .map(|v| v.decode_queue_wait_total_ms.load(Ordering::SeqCst))
        .unwrap_or(0);
    let decode_shed_total = runtime
        .as_ref()
        .map(|v| v.decode_shed_total.load(Ordering::SeqCst))
        .unwrap_or(0);
    let unmapped_ssrc = runtime
        .as_ref()
        .map(|v| v.unmapped_ssrc_activity.load(Ordering::SeqCst))
        .unwrap_or(0);
    let participants = ctx
        .cache
        .guild(guild_id)
        .map(|g| {
            let bot_id = ctx.cache.current_user().id;
            g.voice_states
                .iter()
                .filter(|(uid, vs)| vs.channel_id == Some(voice_channel) && **uid != bot_id)
                .count()
        })
        .unwrap_or(0);

    let hh = elapsed.as_secs() / 3600;
    let mm = (elapsed.as_secs() % 3600) / 60;
    let ss = elapsed.as_secs() % 60;

    let queue_depth = decode_queue_depth();
    let queue_capacity = decode_queue_capacity();
    let decode_failure_pct = {
        let denom = decoded_frames.saturating_add(decode_failures);
        if denom == 0 {
            0.0
        } else {
            (decode_failures as f64 * 100.0) / denom as f64
        }
    };
    let rtf = if decode_audio_total_ms == 0 {
        0.0
    } else {
        decode_total_ms as f64 / decode_audio_total_ms as f64
    };
    let avg_decode_ms = if decode_jobs_total == 0 {
        0.0
    } else {
        decode_total_ms as f64 / decode_jobs_total as f64
    };
    let avg_queue_wait_ms = if decode_jobs_total == 0 {
        0.0
    } else {
        decode_queue_wait_total_ms as f64 / decode_jobs_total as f64
    };
    let decode_shed_per_min = {
        let elapsed_min = (elapsed.as_secs_f64() / 60.0).max(1e-6);
        decode_shed_total as f64 / elapsed_min
    };

    let queue_alert = if queue_depth > 8 {
        "critical"
    } else if queue_depth > 4 {
        "warn"
    } else {
        "ok"
    };
    let rtf_alert = if rtf > 0.8 {
        "critical"
    } else if rtf > 0.5 {
        "warn"
    } else {
        "ok"
    };
    let failure_alert = if decode_failure_pct > 5.0 {
        "critical"
    } else if decode_failure_pct > 1.0 {
        "warn"
    } else {
        "ok"
    };
    let shed_alert = if decode_shed_per_min > 1.0 {
        "critical"
    } else if decode_shed_total > 0 {
        "warn"
    } else {
        "ok"
    };
    let unmapped_alert = if elapsed.as_secs() > 30 && unmapped_ssrc > 0 {
        "warn"
    } else {
        "ok"
    };

    tracing::info!(
        guild = %guild_id,
        queue_depth,
        queue_capacity,
        decode_jobs_total,
        decode_jobs_with_text,
        decode_failure_pct = format!("{decode_failure_pct:.2}"),
        rtf = format!("{rtf:.3}"),
        avg_decode_ms = format!("{avg_decode_ms:.1}"),
        avg_queue_wait_ms = format!("{avg_queue_wait_ms:.1}"),
        decode_shed_per_min = format!("{decode_shed_per_min:.2}"),
        "status metrics snapshot"
    );

    Ok(format!(
        "Transcription status\nVoice channel: <#{}>\nActive for: {hh:02}:{mm:02}:{ss:02}\nParticipants in voice: {}\nQueue depth: {}/{} [{}]\nASR in-flight: {}\nPending commits: {}\nDecode failure: {:.2}% [{}]\nRTF (decode/audio): {:.3} [{}]\nDecode wait/decode ms (avg): {:.1}/{:.1}\nDecode shed: {} ({:.2}/min) [{}]\nUnmapped SSRC: {} [{}]\nTranscript utterances: {}",
        voice_channel.get(),
        participants,
        queue_depth,
        queue_capacity,
        queue_alert,
        inflight,
        pending_commits,
        decode_failure_pct,
        failure_alert,
        rtf,
        rtf_alert,
        avg_queue_wait_ms,
        avg_decode_ms,
        decode_shed_total,
        decode_shed_per_min,
        shed_alert,
        unmapped_ssrc,
        unmapped_alert,
        utterance_count,
    ))
}

pub(super) async fn handle_leave(
    ctx: &Context,
    state: &Arc<AppState>,
    command: &CommandInteraction,
) -> anyhow::Result<String> {
    let guild_id = command.guild_id.context("leave used outside guild")?;

    super::session::finalize_call_for_guild(ctx, state, guild_id).await?;

    Ok("Left voice and finalized transcript.".to_string())
}

pub(super) async fn handle_ask(
    ctx: &Context,
    state: &Arc<AppState>,
    command: &CommandInteraction,
) -> anyhow::Result<String> {
    let guild_id = command.guild_id.context("ask used outside guild")?;

    let question = command
        .data
        .options
        .iter()
        .find(|o| o.name == "question")
        .and_then(|opt| match &opt.value {
            CommandDataOptionValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .context("missing question")?;

    let session_lock = state
        .active_calls
        .get(&guild_id)
        .context("no active call for this guild")?
        .clone();

    let (mut snapshot, started_at) = {
        let session = session_lock.read().await;
        (session.transcript.clone(), session.started_at)
    };

    snapshot.sort_by_key(|u| u.start_ts);

    if snapshot.is_empty() {
        return Ok(
            "No transcribed utterances yet. Try /ask again after someone speaks and pauses briefly."
                .to_string(),
        );
    }

    let transcript = super::summary::format_transcript(ctx, &snapshot, started_at).await;

    let answer = state
        .ai
        .ask(&transcript, &question, None)
        .await
        .unwrap_or_else(|e| format!("ai error: {e}"));

    Ok(answer)
}

pub(super) async fn handle_log(
    ctx: &Context,
    state: &Arc<AppState>,
    command: &CommandInteraction,
) -> anyhow::Result<String> {
    let guild_id = command.guild_id.context("log used outside guild")?;

    let requested_utterances = command
        .data
        .options
        .iter()
        .find(|o| o.name == "utterances")
        .and_then(|opt| match &opt.value {
            CommandDataOptionValue::Integer(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(LOG_DEFAULT_UTTERANCES)
        .clamp(1, 500) as usize;

    let session_lock = state
        .active_calls
        .get(&guild_id)
        .context("no active call for this guild")?
        .clone();

    let (mut snapshot, started_at) = {
        let session = session_lock.read().await;
        (session.transcript.clone(), session.started_at)
    };

    snapshot.sort_by_key(|u| u.start_ts);

    if snapshot.is_empty() {
        return Ok("No transcribed utterances yet.".to_string());
    }

    let start = snapshot
        .len()
        .saturating_sub(requested_utterances);
    let mut transcript =
        super::summary::format_transcript(ctx, &snapshot[start..], started_at).await;

    if transcript.chars().count() > LOG_MAX_DISCORD_CHARS {
        let keep = LOG_MAX_DISCORD_CHARS.saturating_sub(48);
        let tail: String = transcript
            .chars()
            .rev()
            .take(keep)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        transcript = format!("(truncated to recent text)\n{tail}");
    }

    Ok(format!(
        "Recent transcript (last {} utterances):\n{}",
        requested_utterances, transcript
    ))
}

pub(super) async fn handle_autojoin(
    ctx: &Context,
    state: &Arc<AppState>,
    command: &CommandInteraction,
) -> anyhow::Result<String> {
    let guild_id = command.guild_id.context("autojoin used outside guild")?;

    let enabled_opt = command
        .data
        .options
        .iter()
        .find(|o| o.name == "enabled")
        .and_then(|opt| match &opt.value {
            CommandDataOptionValue::Boolean(v) => Some(*v),
            _ => None,
        });

    let mentioned_channel = command
        .data
        .options
        .iter()
        .find(|o| o.name == "channel")
        .and_then(|opt| match &opt.value {
            CommandDataOptionValue::Channel(id) => Some(*id),
            _ => None,
        });

    let voice_channel_id = match mentioned_channel {
        Some(id) => id,
        None => {
            let guild = ctx.cache.guild(guild_id).context("guild not in cache")?;
            guild
                .voice_states
                .get(&command.user.id)
                .and_then(|vs| vs.channel_id)
                .context("you are not connected to a voice channel (or provide a channel mention)")?
        }
    };

    let old_name = {
        let guild = ctx.cache.guild(guild_id).context("guild not in cache")?;
        let chan = guild
            .channels
            .get(&voice_channel_id)
            .context("voice channel not found")?;
        if chan.kind != ChannelType::Voice && chan.kind != ChannelType::Stage {
            anyhow::bail!("autojoin can only be set on voice or stage channels");
        }
        chan.name.clone()
    };

    let configured_suffix = normalized_autojoin_suffix(&state.autojoin_suffix);
    let is_marked = old_name.ends_with(&configured_suffix);
    let enable = enabled_opt.unwrap_or(!is_marked);

    let base_name = strip_known_autojoin_suffix(&old_name, &configured_suffix)
        .unwrap_or_else(|| old_name.clone());

    let new_name = if enable {
        if old_name.ends_with(&configured_suffix) {
            old_name.clone()
        } else {
            let candidate = format!("{base_name}{configured_suffix}");
            if candidate.chars().count() > 100 {
                anyhow::bail!(
                    "cannot enable autojoin because channel name would exceed Discord's 100-character limit"
                );
            }
            candidate
        }
    } else if is_marked {
        base_name
    } else {
        old_name.clone()
    };

    if new_name.trim().is_empty() {
        anyhow::bail!("cannot remove the autojoin suffix because it would leave an empty channel name");
    }

    if new_name != old_name {
        voice_channel_id
            .edit(&ctx.http, serenity::builder::EditChannel::new().name(new_name))
            .await
            .context("failed to rename voice channel (need Manage Channels permission)")?;
    }

    if enable {
        Ok(format!(
            "Autojoin enabled for <#{}>. The bot will auto-start when a non-bot user joins that channel.",
            voice_channel_id.get()
        ))
    } else {
        Ok(format!(
            "Autojoin disabled for <#{}>.",
            voice_channel_id.get()
        ))
    }
}

