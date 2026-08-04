use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use serde::Deserialize;
use serenity::all::{CreateAttachment, CreateMessage, GuildId, UserId, VoiceState};
use serenity::prelude::Context;
use tokio::fs;

use super::super::{AppState, FINALIZE_SETTLE_PASSES, FINALIZE_SETTLE_TIMEOUT, Utterance};
use crate::asr::{clear_unknown_ssrc_audio_for_guild, prune_old_transcripts, should_dispatch_chunk, transcribe_mono_pcm, trim_finalize_tail};

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
    let transcript = load_persisted_transcript(&session.transcript_jsonl_path, session.started_mono).await;

    let transcript_title = super::super::summary::format_call_title(session.started_at);
    let call_duration = session.started_mono.elapsed();
    let should_generate_summary = state.post_call_summary_enabled;
    let include_summary_in_markdown =
        should_generate_summary && state.post_call_summary_include_in_markdown;
    let mut auto_summary = if include_summary_in_markdown {
        super::super::summary::maybe_generate_post_call_summary(ctx, state, &transcript, session.started_at).await
    } else {
        None
    };
    let transcript_text = super::super::summary::format_export_markdown(
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

    let attachment = CreateAttachment::bytes(transcript_text.clone().into_bytes(), filename.clone());
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
            auto_summary = super::super::summary::maybe_generate_post_call_summary(
                ctx,
                state,
                &transcript,
                session.started_at,
            )
            .await;
        }

        if let Some(summary) = auto_summary.as_deref() {
            let summary_message = super::super::summary::format_summary_thread_message(summary);
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

    state.guild_runtimes.remove(&guild_id);

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

async fn settle_and_flush_guild_audio(state: &Arc<AppState>, guild_id: GuildId) {
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

async fn flush_pending_buffers_for_export(
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
            noise_rms_ema = stream.denoiser.noise_rms_ema();
            let entry = &mut stream.buffer;

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
            out.push(Utterance { user_id, start_ts, text });
        }
    }

    out
}

async fn transcribe_finalized_mono_pcm(state: &Arc<AppState>, pcm: Vec<f32>) -> Option<String> {
    transcribe_mono_pcm(Arc::clone(&state.asr), pcm).await
}
