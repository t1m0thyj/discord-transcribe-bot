use std::sync::Arc;

use serenity::all::{Channel, ChannelId, ChannelType, MessageId};
use serenity::prelude::Context;

use super::super::AppState;

const TRANSCRIPT_ATTACHMENT_MAX_BYTES: u64 = 10 * 1024 * 1024;

pub(crate) async fn maybe_load_thread_context(
    ctx: &Context,
    state: &Arc<AppState>,
    channel_id: ChannelId,
) -> anyhow::Result<()> {
    ensure_thread_context_loaded(ctx, state, channel_id).await
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
        .find(|a| a.filename.starts_with("transcript-") && a.filename.ends_with(".md"))
    else {
        return Ok(());
    };

    if attachment.size as u64 > TRANSCRIPT_ATTACHMENT_MAX_BYTES {
        tracing::warn!(
            thread = %channel_id,
            bytes = attachment.size,
            "skipping transcript attachment load: attachment exceeds size cap"
        );
        return Ok(());
    }

    let transcript_text = reqwest::get(&attachment.url).await?.text().await?;
    super::super::upsert_thread_context(state, channel_id, transcript_text);

    tracing::info!(thread = %channel_id, "loaded transcript context from attachment");
    Ok(())
}
