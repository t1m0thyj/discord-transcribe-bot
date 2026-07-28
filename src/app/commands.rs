use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::Context as _;
use chrono::Utc;
use serenity::all::{ChannelId, ChannelType, CommandDataOptionValue, CommandInteraction, GuildId};
use serenity::prelude::Context;
use tokio::sync::{mpsc, RwLock};

use super::{
    AppState, CallSession, INTERACTIVE_DRAIN_TIMEOUT, LOG_DEFAULT_UTTERANCES,
    LOG_MAX_DISCORD_CHARS, Utterance,
};
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
    let text_channel = session.text_channel;
    let utterance_count = session.transcript.len();
    let elapsed = session.started_mono.elapsed();
    drop(session);

    let decoded_frames = state
        .decoded_audio_activity
        .get(&guild_id)
        .map(|v| v.load(Ordering::SeqCst))
        .unwrap_or(0);
    let started_notified = state
        .transcription_started_notified
        .get(&guild_id)
        .map(|v| v.load(Ordering::SeqCst))
        .unwrap_or(false);
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
    let mapped_ssrc = state
        .ssrc_to_user
        .iter()
        .filter(|e| e.key().0 == guild_id)
        .count();
    let buffered_users = state
        .buffers
        .iter()
        .filter(|e| e.key().0 == guild_id && !e.value().pcm.is_empty())
        .count();

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

    Ok(format!(
        "Transcription status\nVoice channel: <#{}>\nText channel: <#{}>\nActive for: {hh:02}:{mm:02}:{ss:02}\nParticipants in voice: {}\nDecoded audio frames seen: {}\nStarted transcribing: {}\nMapped SSRC entries: {}\nUsers with buffered audio: {}\nTranscript utterances: {}\nASR in-flight tasks: {}\nPending transcript commits: {}",
        voice_channel.get(),
        text_channel.get(),
        participants,
        decoded_frames,
        if started_notified { "yes" } else { "no" },
        mapped_ssrc,
        buffered_users,
        utterance_count,
        inflight,
        pending_commits,
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

    let drained =
        super::session::wait_for_capture_quiesce_with_timeout(state, guild_id, INTERACTIVE_DRAIN_TIMEOUT)
            .await;

    let pending = super::session::flush_pending_buffers_for_export(state, guild_id).await;
    if !pending.is_empty() {
        let mut session = session_lock.write().await;
        super::session::upsert_utterances(&mut session.transcript, pending);
    }

    let session = session_lock.read().await;
    if session.transcript.is_empty() {
        return Ok(
            "No transcribed utterances yet. Try /ask again after someone speaks and pauses briefly."
                .to_string(),
        );
    }

    let transcript = super::session::format_transcript(ctx, &session.transcript, session.started_at).await;
    drop(session);

    let answer = crate::gemini::ask_gemini(
        &state.gemini_key,
        &state.gemini_model,
        &transcript,
        &question,
        None,
    )
    .await
    .unwrap_or_else(|e| format!("gemini error: {e}"));

    if drained {
        Ok(answer)
    } else {
        Ok(format!(
            "(Still processing live audio; answer is based on the latest available transcript snapshot.)\n\n{}",
            answer
        ))
    }
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

    let drained =
        super::session::wait_for_capture_quiesce_with_timeout(state, guild_id, INTERACTIVE_DRAIN_TIMEOUT)
            .await;

    let pending = super::session::flush_pending_buffers_for_export(state, guild_id).await;
    if !pending.is_empty() {
        let mut session = session_lock.write().await;
        super::session::upsert_utterances(&mut session.transcript, pending);
    }

    let session = session_lock.read().await;
    if session.transcript.is_empty() {
        return Ok("No transcribed utterances yet.".to_string());
    }

    let start = session
        .transcript
        .len()
        .saturating_sub(requested_utterances);
    let mut transcript =
        super::session::format_transcript(ctx, &session.transcript[start..], session.started_at).await;
    drop(session);

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

    if drained {
        Ok(format!(
            "Recent transcript (last {} utterances):\n{}",
            requested_utterances, transcript
        ))
    } else {
        Ok(format!(
            "Recent transcript (last {} utterances, snapshot while live audio is still processing):\n{}",
            requested_utterances, transcript
        ))
    }
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
    state
        .utterance_senders
        .insert(guild_id, utterance_tx.clone());
    let inflight = Arc::new(AtomicUsize::new(0));
    state.transcription_inflight.insert(guild_id, Arc::clone(&inflight));
    let pending_commits = Arc::new(AtomicUsize::new(0));
    state
        .transcript_pending_commits
        .insert(guild_id, Arc::clone(&pending_commits));
    let decode_activity = Arc::new(AtomicUsize::new(0));
    state
        .decoded_audio_activity
        .insert(guild_id, Arc::clone(&decode_activity));
    let decode_error_activity = Arc::new(AtomicUsize::new(0));
    state
        .decode_error_activity
        .insert(guild_id, Arc::clone(&decode_error_activity));
    let started_notified = Arc::new(AtomicBool::new(false));
    state
        .transcription_started_notified
        .insert(guild_id, Arc::clone(&started_notified));

    let session = Arc::new(RwLock::new(CallSession {
        voice_channel,
        text_channel,
        transcript: Vec::new(),
        started_at: Utc::now(),
        started_mono: Instant::now(),
    }));
    state.active_calls.insert(guild_id, session.clone());

    tokio::spawn(transcript_writer_loop(
        session,
        utterance_rx,
        Arc::clone(&pending_commits),
    ));

    super::session::attach_voice_handlers(
        state,
        super::session::VoiceHandlerAttachContext {
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

    tokio::spawn(super::session::startup_receive_watchdog(
        ctx.clone(),
        Arc::clone(state),
        guild_id,
        voice_channel,
        text_channel,
        0,
    ));

    Ok(())
}
