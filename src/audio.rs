use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use serenity::all::{GuildId, UserId};
use serenity::http::Http;
use songbird::events::{Event, EventContext, EventHandler as VoiceEventHandler};
use tokio::sync::mpsc;

use crate::transcription::{SsrcMap, StreamingDecoderCommand, Streams};

pub struct SpeakingUpdateHandler {
    pub guild_id: GuildId,
    pub ssrc_to_user: Arc<SsrcMap>,
}

#[serenity::async_trait]
impl VoiceEventHandler for SpeakingUpdateHandler {
    async fn act(&self, ctx: &EventContext<'_>) -> Option<Event> {
        if let EventContext::SpeakingStateUpdate(speaking) = ctx {
            if let Some(user_id) = speaking.user_id {
                self.ssrc_to_user
                    .insert((self.guild_id, speaking.ssrc), UserId::new(user_id.0));
            }
        }
        None
    }
}

pub struct ClientDisconnectHandler {
    pub guild_id: GuildId,
    pub ssrc_to_user: Arc<SsrcMap>,
    pub streams: Arc<Streams>,
}

#[serenity::async_trait]
impl VoiceEventHandler for ClientDisconnectHandler {
    async fn act(&self, ctx: &EventContext<'_>) -> Option<Event> {
        let EventContext::ClientDisconnect(disconnect) = ctx else {
            return None;
        };

        let keys: Vec<(GuildId, u32)> = self
            .ssrc_to_user
            .iter()
            .filter_map(|entry| {
                if entry.key().0 == self.guild_id && *entry.value() == UserId::new(disconnect.user_id.0) {
                    Some(*entry.key())
                } else {
                    None
                }
            })
            .collect();

        for key in keys {
            if let Some((_, user_id)) = self.ssrc_to_user.remove(&key) {
                self.streams.remove(&(self.guild_id, user_id));
            }
        }

        None
    }
}

pub struct VoiceTickHandler {
    pub http: Arc<Http>,
    pub text_channel: serenity::all::ChannelId,
    pub voice_channel: serenity::all::ChannelId,
    pub started_notified: Arc<AtomicBool>,
    pub guild_id: GuildId,
    pub ssrc_to_user: Arc<SsrcMap>,
    pub streams: Arc<Streams>,
    pub streaming_decoder_tx: mpsc::Sender<StreamingDecoderCommand>,
    pub enable_denoiser: bool,
    pub decode_activity: Arc<AtomicUsize>,
    pub chunks_accepted_activity: Arc<AtomicUsize>,
    pub decode_failure_activity: Arc<AtomicUsize>,
    pub decoder_queue_dropped: Arc<AtomicUsize>,
    pub unmapped_ssrc_activity: Arc<AtomicUsize>,
}

#[serenity::async_trait]
impl VoiceEventHandler for VoiceTickHandler {
    async fn act(&self, ctx: &EventContext<'_>) -> Option<Event> {
        let EventContext::VoiceTick(tick) = ctx else {
            return None;
        };
        let observed_at = Instant::now();
        let mut heard_users = std::collections::HashSet::<UserId>::new();

        for (ssrc, data) in &tick.speaking {
            if data.decoded_voice.is_none() && data.packet.is_some() {
                self.decode_failure_activity.fetch_add(1, Ordering::SeqCst);
            }

            let Some(decoded) = &data.decoded_voice else {
                continue;
            };

            let Some(user_id) = self
                .ssrc_to_user
                .get(&(self.guild_id, *ssrc))
                .map(|v| *v)
            else {
                self.unmapped_ssrc_activity.fetch_add(1, Ordering::SeqCst);
                continue;
            };

            self.decode_activity.fetch_add(1, Ordering::SeqCst);

            let user_key = (self.guild_id, user_id);
            let mut stream = self.streams.entry(user_key).or_default();
            let pcm_16k = stream
                .denoiser
                .push_stereo_pcm_hybrid(decoded, self.enable_denoiser);
            if pcm_16k.is_empty() {
                continue;
            }

            if self
                .started_notified
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                let http = Arc::clone(&self.http);
                let text_channel = self.text_channel;
                let voice_channel = self.voice_channel;
                tokio::spawn(async move {
                    let _ = text_channel
                        .say(&http, format!("Started transcribing in <#{}>.", voice_channel.get()))
                        .await;
                });
            }

            if self
                .streaming_decoder_tx
                .try_send(StreamingDecoderCommand::AudioChunk {
                    user_id,
                    pcm_16k,
                    observed_at,
                })
                .is_err()
            {
                self.decoder_queue_dropped.fetch_add(1, Ordering::SeqCst);
            } else {
                heard_users.insert(user_id);
                self.chunks_accepted_activity.fetch_add(1, Ordering::SeqCst);
            }
        }

        let _ = self
            .streaming_decoder_tx
            .try_send(StreamingDecoderCommand::TickDone {
                heard_users: heard_users.into_iter().collect(),
                observed_at,
            })
            .map_err(|_| self.decoder_queue_dropped.fetch_add(1, Ordering::SeqCst));

        None
    }
}

