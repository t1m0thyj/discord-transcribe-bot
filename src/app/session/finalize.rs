use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use serenity::all::{CreateAttachment, CreateMessage, GuildId, VoiceState};
use serenity::prelude::Context;

use super::super::AppState;
use crate::app::journal::{load_persisted_transcript, prune_old_transcripts};
use crate::app::summary;
use crate::asr::clear_unknown_ssrc_audio_for_guild;

pub async fn finalize_call_for_guild(
    ctx: &Context,
    state: &Arc<AppState>,
    guild_id: GuildId,
) -> anyhow::Result<()> {
    finalize_call_for_guild_if_current(ctx, state, guild_id, None).await
}

async fn finalize_call_for_guild_if_current(
    ctx: &Context,
    state: &Arc<AppState>,
    guild_id: GuildId,
    expected_session: Option<&Arc<tokio::sync::RwLock<super::super::CallSession>>>,
) -> anyhow::Result<()> {
    let session_start_lock = state
        .session_start_locks
        .entry(guild_id)
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone();
    let _session_start_guard = session_start_lock.lock().await;

    if let Some(expected) = expected_session {
        if !state
            .active_calls
            .get(&guild_id)
            .is_some_and(|current| Arc::ptr_eq(current.value(), expected))
        {
            return Ok(());
        }
        let target_channel = expected.read().await.voice_channel;
        let bot_id = ctx.cache.current_user().id;
        let Some(guild) = ctx.cache.guild(guild_id) else {
            return Ok(());
        };
        if guild
            .voice_states
            .iter()
            .any(|(uid, vs)| vs.channel_id == Some(target_channel) && *uid != bot_id)
        {
            return Ok(());
        }
    }

    let manager = songbird::get(ctx)
        .await
        .context("songbird voice manager unavailable")?
        .clone();
    match tokio::time::timeout(Duration::from_secs(25), manager.remove(guild_id)).await {
        Ok(Ok(())) => {}
        Ok(Err(error)) if manager.get(guild_id).is_some() => return Err(error.into()),
        Ok(Err(error)) => {
            tracing::warn!(guild = %guild_id, "voice remove failed after call disappeared: {error:#}")
        }
        Err(_) if manager.get(guild_id).is_some() => {
            anyhow::bail!("voice remove during finalize timed out")
        }
        Err(_) => {
            tracing::warn!(guild = %guild_id, "voice remove timed out after call disappeared")
        }
    }

    let Some((_gid, session_lock)) = state.active_calls.remove(&guild_id) else {
        return Ok(());
    };

    let runtime = state
        .guild_runtimes
        .get(&guild_id)
        .map(|entry| Arc::clone(entry.value()));
    if let Some(runtime) = &runtime {
        runtime.receive_generation.fetch_add(1, Ordering::SeqCst);
        super::watchdog::detach_and_transcribe_buffers(state, guild_id, runtime);
    }

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

    state.guild_runtimes.remove(&guild_id);
    drop(_session_start_guard);

    // Export work belongs to the detached session. A new /join may now proceed.
    if let Some(runtime) = &runtime {
        let drained = tokio::time::timeout(Duration::from_secs(30), async {
            while runtime.transcription_inflight.load(Ordering::SeqCst) > 0
                || runtime.transcript_pending_commits.load(Ordering::SeqCst) > 0
            {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await;
        if drained.is_err() {
            tracing::warn!(guild = %guild_id, "timed out waiting for old-session transcription; exporting partial transcript");
        }
    }

    let session = session_lock.read().await;
    let transcript =
        load_persisted_transcript(&session.transcript_jsonl_path, session.started_mono).await;

    let transcript_title = summary::format_call_title(session.started_at);
    let call_duration = session.started_mono.elapsed();
    let should_generate_summary = state.post_call_summary_enabled;
    let include_summary_in_markdown =
        should_generate_summary && state.post_call_summary_include_in_markdown;
    let mut auto_summary = if include_summary_in_markdown {
        summary::maybe_generate_post_call_summary(
            ctx,
            state,
            guild_id,
            &transcript,
            session.started_at,
        )
        .await
    } else {
        None
    };
    let transcript_text = summary::format_export_markdown(
        ctx,
        &transcript,
        session.started_at,
        call_duration,
        &transcript_title,
        auto_summary.as_deref(),
        include_summary_in_markdown,
    )
    .await;
    let filename = format!(
        "transcript-{}.md",
        session.started_at.format("%Y%m%d-%H%M%S")
    );

    let local_dir = PathBuf::from("transcripts");

    let attachment =
        CreateAttachment::bytes(transcript_text.clone().into_bytes(), filename.clone());
    let msg = session
        .text_channel
        .send_files(
            &ctx.http,
            vec![attachment],
            CreateMessage::new()
                .content("Call transcript attached. Ask questions or run /summary in the thread."),
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
            let typing = thread.id.start_typing(&ctx.http);
            auto_summary = summary::maybe_generate_post_call_summary(
                ctx,
                state,
                guild_id,
                &transcript,
                session.started_at,
            )
            .await;
            drop(typing);
        }

        if let Some(summary) = auto_summary.as_deref() {
            let summary_message = summary::format_summary_thread_message(summary);
            if let Err(e) = thread.say(&ctx.http, summary_message).await {
                tracing::warn!(guild = %guild_id, "failed to post auto-summary in thread: {e:#}");
            }
        }
    }

    super::super::upsert_thread_context(state, thread.id, transcript_text);

    let prune_dir = local_dir.clone();
    let retention_days = state.transcript_retention_days.max(1);
    tokio::spawn(async move {
        let deleted = prune_old_transcripts(
            &prune_dir,
            Duration::from_secs(retention_days * 24 * 60 * 60),
        )
        .await;
        if deleted > 0 {
            tracing::info!(deleted, "pruned old local transcript files");
        }
    });

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

    finalize_call_for_guild_if_current(ctx, state, guild_id, Some(&session_lock)).await?;

    Ok(())
}
