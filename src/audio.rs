use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serenity::http::Http;
use serenity::all::{GuildId, UserId};
use songbird::events::{Event, EventContext, EventHandler as VoiceEventHandler};
use tokio::sync::mpsc;

use crate::app::Utterance;
use crate::transcription::{
    transcribe_mono_pcm, AsrEngine,
    SILENCE_TICKS_THRESHOLD,
    SsrcMap, Streams,
};

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
            self.ssrc_to_user.remove(&key);
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
    pub enable_denoiser: bool,
    pub utterance_tx: mpsc::Sender<Utterance>,
    pub transcription_inflight: Arc<AtomicUsize>,
    pub transcript_pending_commits: Arc<AtomicUsize>,
    pub decode_shed_total: Arc<AtomicUsize>,
    pub decode_activity: Arc<AtomicUsize>,
    pub decode_failure_activity: Arc<AtomicUsize>,
    pub unmapped_ssrc_activity: Arc<AtomicUsize>,
    pub asr: Arc<AsrEngine>,
    pub live_transcript_debug: bool,
    pub rolling_ingest_max_ms: u64,
    pub rolling_ingest_context_ms: u64,
}

const DECODE_QUEUE_CAPACITY: usize = 8;

struct DecodeJob {
    guild_id: GuildId,
    user_id: UserId,
    start_ts: Instant,
    stage: &'static str,
    pcm: Vec<f32>,
    sample_count: usize,
    asr: Arc<AsrEngine>,
    utterance_tx: mpsc::Sender<Utterance>,
    inflight: Arc<AtomicUsize>,
    pending_commits: Arc<AtomicUsize>,
    decode_shed_total: Arc<AtomicUsize>,
    live_transcript_debug: bool,
}

struct DecodeDispatcher {
    queue: Mutex<VecDeque<DecodeJob>>,
    notify: tokio::sync::Notify,
    capacity: usize,
}

impl DecodeDispatcher {
    fn global() -> Arc<Self> {
        static INSTANCE: OnceLock<Arc<DecodeDispatcher>> = OnceLock::new();
        Arc::clone(INSTANCE.get_or_init(|| {
            let dispatcher = Arc::new(DecodeDispatcher {
                queue: Mutex::new(VecDeque::new()),
                notify: tokio::sync::Notify::new(),
                capacity: DECODE_QUEUE_CAPACITY,
            });
            spawn_decode_worker(Arc::clone(&dispatcher));
            dispatcher
        }))
    }

    fn enqueue(&self, job: DecodeJob) -> Option<DecodeJob> {
        let mut queue = self
            .queue
            .lock()
            .expect("decode queue mutex poisoned");
        let dropped = if queue.len() >= self.capacity {
            queue.pop_front()
        } else {
            None
        };
        queue.push_back(job);
        drop(queue);
        self.notify.notify_one();
        dropped
    }
}

fn spawn_decode_worker(dispatcher: Arc<DecodeDispatcher>) {
    tokio::spawn(async move {
        loop {
            let job = loop {
                if let Some(job) = dispatcher
                    .queue
                    .lock()
                    .expect("decode queue mutex poisoned")
                    .pop_front()
                {
                    break job;
                }

                dispatcher.notify.notified().await;
            };

            process_decode_job(job).await;
        }
    });
}

async fn process_decode_job(job: DecodeJob) {
    if let Some(text) = transcribe_utterance_blocking(&job.asr, job.pcm).await {
        let audio_secs = (job.sample_count as f32 / 16_000.0).max(0.001);
        if let Some(reason) = decode_rejection_reason(&text, audio_secs) {
            tracing::warn!(
                guild = %job.guild_id,
                user = %job.user_id,
                stage = job.stage,
                reason,
                transcript = %text,
                "rejected decoded utterance"
            );
            job.inflight.fetch_sub(1, Ordering::SeqCst);
            return;
        }

        if job.live_transcript_debug {
            tracing::debug!(
                user = %job.user_id,
                transcript = %text,
                stage = job.stage,
                "final transcription"
            );
        }

        job.pending_commits.fetch_add(1, Ordering::SeqCst);
        if job
            .utterance_tx
            .send(Utterance {
                user_id: job.user_id,
                start_ts: job.start_ts,
                text,
            })
            .await
            .is_err()
        {
            job.pending_commits.fetch_sub(1, Ordering::SeqCst);
        }
    }

    job.inflight.fetch_sub(1, Ordering::SeqCst);
}

fn queue_decode_job(job: DecodeJob) {
    let dispatcher = DecodeDispatcher::global();
    if let Some(dropped) = dispatcher.enqueue(job) {
        dropped.decode_shed_total.fetch_add(1, Ordering::SeqCst);
        dropped.inflight.fetch_sub(1, Ordering::SeqCst);
        tracing::warn!(
            guild = %dropped.guild_id,
            user = %dropped.user_id,
            "decode queue full; dropped oldest queued chunk"
        );
    }
}

fn should_dispatch_chunk(pcm: &[f32], voiced_ticks: u32, noise_rms_ema: f32) -> bool {
    if voiced_ticks < 12 {
        return false;
    }

    let rms = compute_rms(pcm);
    let floor = (noise_rms_ema * 1.4).max(0.0025);
    rms >= floor
}

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn decode_rejection_reason(text: &str, audio_secs: f32) -> Option<&'static str> {
    let normalized = text
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || c.is_ascii_whitespace())
        .collect::<String>();

    const HALLUCINATION_BLOCKLIST: &[&str] = &[
        "thank you",
        "thanks for watching",
        "thank you for watching",
        "subtitles by",
        "captions by",
    ];

    if HALLUCINATION_BLOCKLIST
        .iter()
        .any(|phrase| normalized == *phrase)
    {
        return Some("blocklist");
    }

    let chars_per_second = text.chars().count() as f32 / audio_secs;
    if chars_per_second > 25.0 {
        return Some("implausible_char_rate");
    }

    None
}

#[serenity::async_trait]
impl VoiceEventHandler for VoiceTickHandler {
    async fn act(&self, ctx: &EventContext<'_>) -> Option<Event> {
        let EventContext::VoiceTick(tick) = ctx else {
            return None;
        };

        let mut currently_speaking = std::collections::HashSet::<UserId>::new();
        let max_ingest_samples =
            ((self.rolling_ingest_max_ms as usize) * 16).max(1_600);
        let base_keep_context_samples = (self.rolling_ingest_context_ms as usize) * 16;

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
                // Decoded audio without an SSRC mapping is not usable yet.
                // Count it as startup receive failure signal so watchdog can recover.
                self.unmapped_ssrc_activity.fetch_add(1, Ordering::SeqCst);
                continue;
            };

            self.decode_activity.fetch_add(1, Ordering::SeqCst);

            let user_key = (self.guild_id, user_id);

            let mut stream = self.streams.entry(user_key).or_default();
            let processed = stream
                .denoiser
                .push_stereo_pcm(decoded, self.enable_denoiser);
            if processed.speech_active {
                currently_speaking.insert(user_id);
            }
            if processed.pcm_16k.is_empty() {
                continue;
            }

            if stream.buffer.utterance_start.is_none() && !processed.speech_active {
                continue;
            }

            if processed.speech_active {
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
            }

            let cleaned = processed.pcm_16k;
            let noise_rms_ema = stream.denoiser.noise_rms_ema();

            let entry = &mut stream.buffer;
            if entry.utterance_start.is_none() {
                entry.utterance_start = Some(Instant::now());
            }
            if processed.speech_active {
                entry.silent_ticks = 0;
            }
            entry.pcm.extend_from_slice(&cleaned);

            if processed.speech_active {
                entry.voiced_ticks = entry.voiced_ticks.saturating_add(1);
            }

            let mut maybe_rollover_final = None;
            if entry.pcm.len() >= max_ingest_samples {
                let max_keep_without_starving = max_ingest_samples.saturating_sub(1_600);
                let keep = base_keep_context_samples
                    .min(max_keep_without_starving)
                    .min(entry.pcm.len());
                let split_at = choose_rollover_split_index(
                    &entry.pcm,
                    keep,
                    max_keep_without_starving,
                );
                if split_at >= 1_600 {
                    let start_ts = entry.utterance_start.take().unwrap_or_else(Instant::now);
                    let source_len = entry.pcm.len();

                    let tail = entry.pcm.split_off(split_at);
                    let chunk = std::mem::replace(&mut entry.pcm, tail);
                    let chunk_voiced = if source_len == 0 {
                        0
                    } else {
                        ((entry.voiced_ticks as u64 * split_at as u64) / source_len as u64) as u32
                    };
                    entry.voiced_ticks = entry.voiced_ticks.saturating_sub(chunk_voiced);

                    entry.utterance_start = Some(
                        Instant::now()
                            .checked_sub(Duration::from_millis(self.rolling_ingest_context_ms))
                            .unwrap_or_else(Instant::now),
                    );

                    maybe_rollover_final = Some((start_ts, chunk, chunk_voiced, noise_rms_ema));
                }
            }

            tracing::trace!(
                guild = %self.guild_id,
                user = %user_id,
                ssrc = %ssrc,
                samples = cleaned.len(),
                "processed 16k pcm tick"
            );

            if let Some((start_ts, pcm, voiced_ticks, noise_rms_ema)) = maybe_rollover_final {
                if !should_dispatch_chunk(&pcm, voiced_ticks, noise_rms_ema) {
                    continue;
                }
                self.transcription_inflight.fetch_add(1, Ordering::SeqCst);
                let sample_count = pcm.len();
                queue_decode_job(DecodeJob {
                    guild_id: self.guild_id,
                    user_id,
                    start_ts,
                    stage: "rollover",
                    pcm,
                    sample_count,
                    asr: Arc::clone(&self.asr),
                    utterance_tx: self.utterance_tx.clone(),
                    inflight: Arc::clone(&self.transcription_inflight),
                    pending_commits: Arc::clone(&self.transcript_pending_commits),
                    decode_shed_total: Arc::clone(&self.decode_shed_total),
                    live_transcript_debug: self.live_transcript_debug,
                });
            }
        }

        let tracked_users: Vec<(GuildId, UserId)> =
            self.streams.iter().map(|e| *e.key()).collect();
        for user_key in tracked_users {
            if user_key.0 != self.guild_id {
                continue;
            }
            let user_id = user_key.1;
            if currently_speaking.contains(&user_id) {
                continue;
            }

            let mut maybe_job = None;
            if let Some(mut stream) = self.streams.get_mut(&user_key) {
                let should_finalize = {
                    let entry = &mut stream.buffer;
                    entry.silent_ticks = entry.silent_ticks.saturating_add(1);
                    entry.silent_ticks >= SILENCE_TICKS_THRESHOLD && !entry.pcm.is_empty()
                };

                if !should_finalize {
                    continue;
                }

                let flushed = if self.enable_denoiser {
                    stream.denoiser.flush_pending()
                } else {
                    Vec::new()
                };

                let entry = &mut stream.buffer;
                entry.pcm.extend(flushed);
                let start_ts = entry.utterance_start.take().unwrap_or_else(Instant::now);
                let pcm = std::mem::take(&mut entry.pcm);
                let voiced_ticks = std::mem::take(&mut entry.voiced_ticks);
                entry.silent_ticks = 0;
                let noise_rms_ema = stream.denoiser.noise_rms_ema();
                maybe_job = Some((start_ts, pcm, voiced_ticks, noise_rms_ema));
            }

            if let Some((start_ts, pcm, voiced_ticks, noise_rms_ema)) = maybe_job {
                if !should_dispatch_chunk(&pcm, voiced_ticks, noise_rms_ema) {
                    continue;
                }
                self.transcription_inflight.fetch_add(1, Ordering::SeqCst);
                let sample_count = pcm.len();
                queue_decode_job(DecodeJob {
                    guild_id: self.guild_id,
                    user_id,
                    start_ts,
                    stage: "silence",
                    pcm,
                    sample_count,
                    asr: Arc::clone(&self.asr),
                    utterance_tx: self.utterance_tx.clone(),
                    inflight: Arc::clone(&self.transcription_inflight),
                    pending_commits: Arc::clone(&self.transcript_pending_commits),
                    decode_shed_total: Arc::clone(&self.decode_shed_total),
                    live_transcript_debug: self.live_transcript_debug,
                });
            }
        }

        None
    }
}

fn choose_rollover_split_index(
    pcm: &[f32],
    keep_samples: usize,
    max_keep_without_starving: usize,
) -> usize {
    let len = pcm.len();
    if len < 3_200 {
        return len.saturating_sub(keep_samples);
    }

    let min_split = len.saturating_sub(max_keep_without_starving).max(1_600);
    let max_split = len.saturating_sub(1_600);
    if min_split >= max_split {
        return len.saturating_sub(keep_samples).clamp(min_split, max_split);
    }

    let nominal = len.saturating_sub(keep_samples).clamp(min_split, max_split);
    let search_radius = (keep_samples / 2).clamp(800, 8_000);
    let search_start = nominal.saturating_sub(search_radius).max(min_split);
    let search_end = nominal.saturating_add(search_radius).min(max_split);

    let frame = 800usize; // 50 ms at 16 kHz
    let hop = 160usize; // 10 ms at 16 kHz
    let mut best = nominal;
    let mut best_rms = f32::INFINITY;

    let mut idx = search_start;
    while idx <= search_end {
        let left = idx.saturating_sub(frame / 2);
        let right = (left + frame).min(len);
        let left = right.saturating_sub(frame);
        if right > left {
            let window = &pcm[left..right];
            let rms = (window.iter().map(|s| s * s).sum::<f32>() / window.len() as f32).sqrt();
            if rms < best_rms {
                best_rms = rms;
                best = idx;
            }
        }

        let next = idx.saturating_add(hop);
        if next <= idx {
            break;
        }
        idx = next;
    }

    best.clamp(min_split, max_split)
}

async fn transcribe_utterance_blocking(
    asr: &Arc<AsrEngine>,
    pcm_mono: Vec<f32>,
) -> Option<String> {
    transcribe_mono_pcm(Arc::clone(asr), pcm_mono).await
}

