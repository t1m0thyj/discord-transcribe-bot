use std::sync::Arc;

use serenity::all::{ChannelId, ChannelType, GuildId, VoiceState};
use serenity::prelude::Context;

use super::AppState;

pub async fn maybe_autojoin_on_voice_state(
    ctx: &Context,
    state: &Arc<AppState>,
    guild_id: GuildId,
    old: Option<VoiceState>,
    new: VoiceState,
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

        state
            .autojoin_text_channel_override
            .or_else(|| pick_autojoin_text_channel(&guild, channel_id))
    }) else {
        tracing::warn!(
            guild = %guild_id,
            "autojoin skipped: no suitable text channel found"
        );
        return;
    };

    if let Err(e) =
        super::session::start_call_session(ctx, state, guild_id, channel_id, text_channel_id).await
    {
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

pub(super) fn pick_autojoin_text_channel(
    guild: &serenity::model::guild::Guild,
    voice_channel_id: ChannelId,
) -> Option<ChannelId> {
    let channels: Vec<ChannelSummary> = guild
        .channels
        .iter()
        .map(|(id, channel)| ChannelSummary {
            id: *id,
            kind: channel.kind,
            parent_id: channel.parent_id,
            position: channel.position,
        })
        .collect();

    pick_autojoin_text_channel_from_channels(&channels, voice_channel_id, guild.system_channel_id)
}

#[derive(Clone, Copy)]
struct ChannelSummary {
    id: ChannelId,
    kind: ChannelType,
    parent_id: Option<ChannelId>,
    position: u16,
}

fn pick_autojoin_text_channel_from_channels(
    channels: &[ChannelSummary],
    voice_channel_id: ChannelId,
    system_channel_id: Option<ChannelId>,
) -> Option<ChannelId> {
    let voice_parent_id = channels
        .iter()
        .find(|channel| channel.id == voice_channel_id)
        .and_then(|channel| channel.parent_id);

    if let Some(parent_id) = voice_parent_id {
        let same_category = channels
            .iter()
            .filter(|channel| {
                channel.kind == ChannelType::Text && channel.parent_id == Some(parent_id)
            })
            .min_by_key(|channel| (channel.position, channel.id.get()))
            .map(|channel| channel.id);

        if same_category.is_some() {
            return same_category;
        }
    }

    if let Some(system_id) = system_channel_id {
        return Some(system_id);
    }

    channels
        .iter()
        .filter(|channel| channel.kind == ChannelType::Text)
        .min_by_key(|channel| (channel.position, channel.id.get()))
        .map(|channel| channel.id)
}

pub(super) fn normalized_autojoin_suffix(suffix: &str) -> String {
    let trimmed = suffix.trim();
    if trimmed.is_empty() {
        return " [Transcribe]".to_string();
    }
    format!(" {trimmed}")
}

pub(super) fn strip_known_autojoin_suffix(name: &str, suffix: &str) -> Option<String> {
    name.strip_suffix(suffix).map(ToString::to_string)
}

#[cfg(test)]
mod tests {
    use serenity::all::{ChannelId, ChannelType};

    use super::{
        normalized_autojoin_suffix, pick_autojoin_text_channel_from_channels,
        strip_known_autojoin_suffix, ChannelSummary,
    };

    fn channel(
        id: u64,
        kind: ChannelType,
        parent_id: Option<u64>,
        position: u16,
    ) -> ChannelSummary {
        ChannelSummary {
            id: ChannelId::new(id),
            kind,
            parent_id: parent_id.map(ChannelId::new),
            position,
        }
    }

    #[test]
    fn normalized_suffix_uses_default_when_blank() {
        assert_eq!(normalized_autojoin_suffix("   "), " [Transcribe]");
    }

    #[test]
    fn normalized_suffix_adds_leading_space() {
        assert_eq!(normalized_autojoin_suffix("[Live]"), " [Live]");
    }

    #[test]
    fn strip_known_suffix_only_when_present() {
        assert_eq!(
            strip_known_autojoin_suffix("General [Transcribe]", " [Transcribe]"),
            Some("General".to_string())
        );
        assert_eq!(
            strip_known_autojoin_suffix("General", " [Transcribe]"),
            None
        );
    }

    #[test]
    fn suffix_helpers_handle_leading_space_and_suffix_only_name() {
        assert_eq!(normalized_autojoin_suffix(" [Live]"), " [Live]");
        assert_eq!(
            strip_known_autojoin_suffix(" [Transcribe]", " [Transcribe]"),
            Some(String::new())
        );
    }

    #[test]
    fn autojoin_text_channel_prefers_category_then_system_then_global_tiebreak() {
        let channels = [
            channel(1, ChannelType::Voice, Some(10), 0),
            channel(4, ChannelType::Text, Some(10), 2),
            channel(3, ChannelType::Text, Some(10), 2),
            channel(2, ChannelType::Text, None, 0),
        ];
        assert_eq!(
            pick_autojoin_text_channel_from_channels(
                &channels,
                ChannelId::new(1),
                Some(ChannelId::new(2))
            ),
            Some(ChannelId::new(3))
        );

        let no_category_text = [
            channel(1, ChannelType::Voice, Some(10), 0),
            channel(2, ChannelType::Text, None, 3),
        ];
        assert_eq!(
            pick_autojoin_text_channel_from_channels(
                &no_category_text,
                ChannelId::new(1),
                Some(ChannelId::new(99))
            ),
            Some(ChannelId::new(99))
        );
        assert_eq!(
            pick_autojoin_text_channel_from_channels(&no_category_text, ChannelId::new(1), None),
            Some(ChannelId::new(2))
        );
    }
}
