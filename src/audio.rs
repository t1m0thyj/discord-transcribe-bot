use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use serenity::http::Http;
use serenity::all::{GuildId, UserId};
use songbird::events::{Event, EventContext, EventHandler as VoiceEventHandler};

use crate::app::{GuildRuntime, Utterance};
use crate::denoiser::compute_rms;
use crate::transcription::{
    should_dispatch_chunk, transcribe_mono_pcm, AsrEngine,
    trim_finalize_tail,
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
    pub runtime: Arc<GuildRuntime>,
    pub guild_id: GuildId,
    pub ssrc_to_user: Arc<SsrcMap>,
    pub streams: Arc<Streams>,
    pub enable_denoiser: bool,
    pub asr: Arc<AsrEngine>,
    pub live_transcript_debug: bool,
    pub silence_ticks_threshold: u32,
    pub rolling_ingest_max_ms: u64,
    pub rolling_ingest_context_ms: u64,
}

const DECODE_QUEUE_CAPACITY: usize = 8;
const UNKNOWN_SSRC_MAX_TRACKED: usize = 8;
const UNKNOWN_SSRC_MAX_SAMPLES: usize = 96_000;
const UNKNOWN_SSRC_RETENTION: Duration = Duration::from_secs(1);

struct UnknownSsrcAudio {
    samples: VecDeque<i16>,
    last_update: Instant,
}

impl UnknownSsrcAudio {
    fn new() -> Self {
        Self {
            samples: VecDeque::new(),
            last_update: Instant::now(),
        }
    }

    fn push(&mut self, decoded: &[i16]) {
        self.last_update = Instant::now();
        self.samples.extend(decoded.iter().copied());
        let overflow = self.samples.len().saturating_sub(UNKNOWN_SSRC_MAX_SAMPLES);
        if overflow > 0 {
            self.samples.drain(..overflow);
        }
    }

    fn drain_vec(&mut self) -> Vec<i16> {
        self.last_update = Instant::now();
        self.samples.drain(..).collect()
    }
}

fn unknown_ssrc_buffers() -> &'static Mutex<HashMap<(GuildId, u32), UnknownSsrcAudio>> {
    static UNKNOWN_SSRC_BUFFERS: OnceLock<Mutex<HashMap<(GuildId, u32), UnknownSsrcAudio>>> = OnceLock::new();
    UNKNOWN_SSRC_BUFFERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn push_unknown_ssrc_audio(guild_id: GuildId, ssrc: u32, decoded: &[i16]) {
    let now = Instant::now();
    let mut map = unknown_ssrc_buffers()
        .lock()
        .expect("unknown ssrc buffer mutex poisoned");

    map.retain(|_, buf| now.saturating_duration_since(buf.last_update) <= UNKNOWN_SSRC_RETENTION);

    if !map.contains_key(&(guild_id, ssrc)) && map.len() >= UNKNOWN_SSRC_MAX_TRACKED {
        return;
    }

    let entry = map.entry((guild_id, ssrc)).or_insert_with(UnknownSsrcAudio::new);
    entry.push(decoded);
}

fn take_unknown_ssrc_audio(guild_id: GuildId, ssrc: u32) -> Vec<i16> {
    let mut map = unknown_ssrc_buffers()
        .lock()
        .expect("unknown ssrc buffer mutex poisoned");
    map.remove(&(guild_id, ssrc))
        .map(|mut buf| buf.drain_vec())
        .unwrap_or_default()
}

pub fn clear_unknown_ssrc_audio_for_guild(guild_id: GuildId) {
    let mut map = unknown_ssrc_buffers()
        .lock()
        .expect("unknown ssrc buffer mutex poisoned");
    map.retain(|(g, _), _| *g != guild_id);
}

struct DecodeJob {
    guild_id: GuildId,
    user_id: UserId,
    start_ts: Instant,
    stage: &'static str,
    pcm: Vec<f32>,
    enqueued_at: Instant,
    runtime: Arc<GuildRuntime>,
    asr: Arc<AsrEngine>,
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

pub fn decode_queue_depth() -> usize {
    let dispatcher = DecodeDispatcher::global();
    let queue = dispatcher
        .queue
        .lock()
        .expect("decode queue mutex poisoned");
    queue.len()
}

pub fn decode_queue_capacity() -> usize {
    DECODE_QUEUE_CAPACITY
}

fn spawn_decode_worker(dispatcher: Arc<DecodeDispatcher>) {
    tokio::spawn(async move {
        loop {
            let job = loop {
                let maybe_job = {
                    let mut queue = dispatcher
                        .queue
                        .lock()
                        .expect("decode queue mutex poisoned");
                    queue.pop_front()
                };

                if let Some(job) = maybe_job {
                    break job;
                }

                dispatcher.notify.notified().await;
            };

            process_decode_job(job).await;
        }
    });
}

async fn process_decode_job(job: DecodeJob) {
    let queue_wait_ms = job.enqueued_at.elapsed().as_millis() as usize;
    let audio_ms = (job.pcm.len().saturating_mul(1000)) / 16_000;
    let decode_started = Instant::now();
    let decode_result = transcribe_utterance_blocking(&job.asr, job.pcm).await;
    let decode_ms = decode_started.elapsed().as_millis() as usize;

    job.runtime.decode_jobs_total.fetch_add(1, Ordering::SeqCst);
    job.runtime
        .decode_audio_total_ms
        .fetch_add(audio_ms, Ordering::SeqCst);
    job.runtime
        .decode_total_ms
        .fetch_add(decode_ms, Ordering::SeqCst);
    job.runtime
        .decode_queue_wait_total_ms
        .fetch_add(queue_wait_ms, Ordering::SeqCst);
    job.runtime
        .decode_last_ms
        .store(decode_ms, Ordering::SeqCst);
    job.runtime
        .decode_queue_wait_last_ms
        .store(queue_wait_ms, Ordering::SeqCst);

    if let Some(text) = decode_result {
        if job.live_transcript_debug {
            tracing::debug!(
                user = %job.user_id,
                transcript = %text,
                stage = job.stage,
                "final transcription"
            );
        }

        job.runtime
            .decode_jobs_with_text
            .fetch_add(1, Ordering::SeqCst);

        job.runtime
            .transcript_pending_commits
            .fetch_add(1, Ordering::SeqCst);
        if job
            .runtime
            .utterance_tx
            .send(Utterance {
                user_id: job.user_id,
                start_ts: job.start_ts,
                text,
            })
            .await
            .is_err()
        {
            job.runtime
                .transcript_pending_commits
                .fetch_sub(1, Ordering::SeqCst);
        }
    }

    job.runtime
        .transcription_inflight
        .fetch_sub(1, Ordering::SeqCst);
}

fn queue_decode_job(job: DecodeJob) {
    let dispatcher = DecodeDispatcher::global();
    if let Some(dropped) = dispatcher.enqueue(job) {
        dropped
            .runtime
            .decode_shed_total
            .fetch_add(1, Ordering::SeqCst);
        dropped
            .runtime
            .transcription_inflight
            .fetch_sub(1, Ordering::SeqCst);
        tracing::warn!(
            guild = %dropped.guild_id,
            user = %dropped.user_id,
            "decode queue full; dropped oldest queued chunk"
        );
    }
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
                self.runtime
                    .decode_failure_activity
                    .fetch_add(1, Ordering::SeqCst);
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
                self.runtime
                    .unmapped_ssrc_activity
                    .fetch_add(1, Ordering::SeqCst);
                push_unknown_ssrc_audio(self.guild_id, *ssrc, decoded);
                continue;
            };

            self.runtime
                .decoded_audio_activity
                .fetch_add(1, Ordering::SeqCst);

            let user_key = (self.guild_id, user_id);

            let mut stream = self.streams.entry(user_key).or_default();
            let mut replay = take_unknown_ssrc_audio(self.guild_id, *ssrc);
            let processed = if replay.is_empty() {
                stream.denoiser.push_stereo_pcm(decoded, self.enable_denoiser)
            } else {
                replay.extend_from_slice(decoded);
                stream
                    .denoiser
                    .push_stereo_pcm(&replay, self.enable_denoiser)
            };
            let resample_errors = stream.denoiser.take_resample_error_count();
            if resample_errors > 0 {
                self.runtime
                    .resample_error_total
                    .fetch_add(resample_errors, Ordering::SeqCst);
            }
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
                    .runtime
                    .transcription_started_notified
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
                if let Err(rejection) = should_dispatch_chunk(&pcm, voiced_ticks, noise_rms_ema) {
                    self.runtime
                        .dispatch_gate_total
                        .fetch_add(1, Ordering::SeqCst);
                    tracing::debug!(
                        guild = %self.guild_id,
                        user = %user_id,
                        stage = "rollover",
                        reason = rejection.reason,
                        voiced_ticks = rejection.voiced_ticks,
                        rms = rejection.rms,
                        floor = rejection.floor,
                        "dispatch gate rejected utterance"
                    );
                    continue;
                }
                self.runtime
                    .transcription_inflight
                    .fetch_add(1, Ordering::SeqCst);
                queue_decode_job(DecodeJob {
                    guild_id: self.guild_id,
                    user_id,
                    start_ts,
                    stage: "rollover",
                    pcm,
                    enqueued_at: Instant::now(),
                    runtime: Arc::clone(&self.runtime),
                    asr: Arc::clone(&self.asr),
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
                    entry.silent_ticks >= self.silence_ticks_threshold && !entry.pcm.is_empty()
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
                let mut pcm = std::mem::take(&mut entry.pcm);
                let voiced_ticks = std::mem::take(&mut entry.voiced_ticks);
                let final_silent_ticks = entry.silent_ticks;
                entry.silent_ticks = 0;
                trim_finalize_tail(&mut pcm, final_silent_ticks);
                let noise_rms_ema = stream.denoiser.noise_rms_ema();
                maybe_job = Some((start_ts, pcm, voiced_ticks, noise_rms_ema));
            }

            if let Some((start_ts, pcm, voiced_ticks, noise_rms_ema)) = maybe_job {
                if let Err(rejection) = should_dispatch_chunk(&pcm, voiced_ticks, noise_rms_ema) {
                    self.runtime
                        .dispatch_gate_total
                        .fetch_add(1, Ordering::SeqCst);
                    tracing::debug!(
                        guild = %self.guild_id,
                        user = %user_id,
                        stage = "silence",
                        reason = rejection.reason,
                        voiced_ticks = rejection.voiced_ticks,
                        rms = rejection.rms,
                        floor = rejection.floor,
                        "dispatch gate rejected utterance"
                    );
                    continue;
                }
                self.runtime
                    .transcription_inflight
                    .fetch_add(1, Ordering::SeqCst);
                queue_decode_job(DecodeJob {
                    guild_id: self.guild_id,
                    user_id,
                    start_ts,
                    stage: "silence",
                    pcm,
                    enqueued_at: Instant::now(),
                    runtime: Arc::clone(&self.runtime),
                    asr: Arc::clone(&self.asr),
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
            let rms = compute_rms(window);
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

