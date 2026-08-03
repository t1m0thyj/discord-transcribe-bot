use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context as _;
use chrono::Utc;
use serenity::all::{ChannelId, ChannelType, CommandDataOptionValue, CommandInteraction, GuildId};
use serenity::prelude::Context;
use tokio::fs;
use tokio::sync::{mpsc, RwLock};

use super::{
    AppState, CallSession, GuildRuntime, LOG_DEFAULT_UTTERANCES,
    LOG_MAX_DISCORD_CHARS, Utterance,
};
use crate::audio::{decode_queue_capacity, decode_queue_depth};
use crate::transcription::transcript_writer_loop;

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

    start_call_session(ctx, state, guild_id, channel_id, command.channel_id).await?;

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

pub async fn maybe_autojoin_on_voice_state(
    ctx: &Context,
    state: &Arc<AppState>,
    guild_id: GuildId,
    old: Option<serenity::all::VoiceState>,
    new: serenity::all::VoiceState,
) {
    if state.active_calls.contains_key(&guild_id) {
        return;
    }

    let Some(channel_id) = new.channel_id else {
        return;
    };

    if old.as_ref().and_then(|v| v.channel_id) == Some(channel_id) {
        return;
    }

    let bot_id = ctx.cache.current_user().id;
    if new.user_id == bot_id {
        return;
    }

    let Some(text_channel_id) = ({
        let Some(guild) = ctx.cache.guild(guild_id) else {
            return;
        };

        let Some(target_channel) = guild.channels.get(&channel_id) else {
            return;
        };

        if target_channel.kind != ChannelType::Voice && target_channel.kind != ChannelType::Stage {
            return;
        }

        let configured_suffix = normalized_autojoin_suffix(&state.autojoin_suffix);
        if !target_channel.name.ends_with(&configured_suffix) {
            return;
        }

        pick_autojoin_text_channel(&guild, channel_id)
    }) else {
        tracing::warn!(
            guild = %guild_id,
            "autojoin skipped: no suitable text channel found"
        );
        return;
    };

    if let Err(e) = start_call_session(ctx, state, guild_id, channel_id, text_channel_id).await {
        tracing::warn!(guild = %guild_id, "autojoin failed to start session: {e:#}");
        return;
    }

    let _ = text_channel_id
        .say(
            &ctx.http,
            format!(
                "Autojoined <#{}> because the channel is marked with {}.",
                channel_id.get(),
                normalized_autojoin_suffix(&state.autojoin_suffix)
            ),
        )
        .await;
}

fn pick_autojoin_text_channel(
    guild: &serenity::model::guild::Guild,
    voice_channel_id: ChannelId,
) -> Option<ChannelId> {
    let voice_parent_id = guild
        .channels
        .get(&voice_channel_id)
        .and_then(|ch| ch.parent_id);

    if let Some(parent_id) = voice_parent_id {
        let same_category = guild
            .channels
            .iter()
            .filter(|(_, ch)| ch.kind == ChannelType::Text && ch.parent_id == Some(parent_id))
            .min_by_key(|(_, ch)| (ch.position, ch.id.get()))
            .map(|(id, _)| *id);

        if same_category.is_some() {
            return same_category;
        }
    }

    if let Some(system_id) = guild.system_channel_id {
        return Some(system_id);
    }

    guild
        .channels
        .iter()
        .filter(|(_, ch)| ch.kind == ChannelType::Text)
        .min_by_key(|(_, ch)| (ch.position, ch.id.get()))
        .map(|(id, _)| *id)
}

fn normalized_autojoin_suffix(suffix: &str) -> String {
    let trimmed = suffix.trim();
    if trimmed.is_empty() {
        return " [Transcribe]".to_string();
    }
    format!(" {trimmed}")
}

fn strip_known_autojoin_suffix(name: &str, suffix: &str) -> Option<String> {
    name.strip_suffix(suffix).map(ToString::to_string)
}

pub(super) async fn start_call_session(
    ctx: &Context,
    state: &Arc<AppState>,
    guild_id: GuildId,
    voice_channel: ChannelId,
    text_channel: ChannelId,
) -> anyhow::Result<()> {
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
    fs::File::create(&transcript_jsonl_path).await.with_context(|| {
        format!(
            "failed to create transcript journal file {}",
            transcript_jsonl_path.display()
        )
    })?;

    let runtime = Arc::new(GuildRuntime {
        utterance_tx: utterance_tx.clone(),
        transcription_inflight: Arc::new(AtomicUsize::new(0)),
        transcript_pending_commits: Arc::new(AtomicUsize::new(0)),
        decode_jobs_total: Arc::new(AtomicUsize::new(0)),
        decode_jobs_with_text: Arc::new(AtomicUsize::new(0)),
        decode_audio_total_ms: Arc::new(AtomicUsize::new(0)),
        decode_total_ms: Arc::new(AtomicUsize::new(0)),
        decode_queue_wait_total_ms: Arc::new(AtomicUsize::new(0)),
        decode_last_ms: Arc::new(AtomicUsize::new(0)),
        decode_queue_wait_last_ms: Arc::new(AtomicUsize::new(0)),
        decode_shed_total: Arc::new(AtomicUsize::new(0)),
        dispatch_gate_total: Arc::new(AtomicUsize::new(0)),
        resample_error_total: Arc::new(AtomicUsize::new(0)),
        decoded_audio_activity: Arc::new(AtomicUsize::new(0)),
        decode_failure_activity: Arc::new(AtomicUsize::new(0)),
        unmapped_ssrc_activity: Arc::new(AtomicUsize::new(0)),
        transcription_started_notified: Arc::new(AtomicBool::new(false)),
        recovery_lock: Arc::new(tokio::sync::Mutex::new(())),
    });
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

    tokio::spawn(transcript_writer_loop(
        session,
        utterance_rx,
        Arc::clone(&runtime.transcript_pending_commits),
        transcript_jsonl_path,
    ));

    super::session::attach_voice_handlers(
        state,
        super::session::VoiceHandlerAttachContext {
            http: Arc::clone(&ctx.http),
            guild_id,
            text_channel,
            voice_channel,
            call_lock: Arc::clone(&call_lock),
            runtime,
        },
    )
    .await;

    tokio::spawn(super::session::startup_receive_watchdog(
        ctx.clone(),
        Arc::clone(state),
        guild_id,
        voice_channel,
        text_channel,
        0,
    ));

    tokio::spawn(super::session::steady_state_receive_watchdog(
        ctx.clone(),
        Arc::clone(state),
        guild_id,
        voice_channel,
        text_channel,
    ));

    Ok(())
}
