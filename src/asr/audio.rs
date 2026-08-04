use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serenity::all::{GuildId, UserId};
use serenity::http::Http;
use songbird::events::{Event, EventContext, EventHandler as VoiceEventHandler};

use crate::app::GuildRuntime;

use super::decoder::{queue_decode_job, DecodeJob};
use super::frontend::{ProcessedSpeechChunk, compute_rms};
use super::pipeline::{
    UserAudioBuffer, should_dispatch_chunk, trim_finalize_tail, AsrEngine, SsrcMap, Streams,
};

const UNKNOWN_SSRC_MAX_TRACKED: usize = 8;
const UNKNOWN_SSRC_MAX_SAMPLES: usize = 96_000;
const UNKNOWN_SSRC_RETENTION: Duration = Duration::from_secs(1);

#[derive(Clone, Copy)]
struct IngestParams {
    max_ingest_samples: usize,
    keep_context_samples: usize,
    silence_ticks_threshold: u32,
}

enum IngestOutcome {
    None,
    Rollover {
        start_ts: Instant,
        pcm: Vec<f32>,
        voiced_ticks: u32,
    },
    Endpoint {
        start_ts: Instant,
        pcm: Vec<f32>,
        voiced_ticks: u32,
        silent_ticks: u32,
    },
}

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

pub use super::decoder::{decode_queue_capacity, decode_queue_depth};

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

#[serenity::async_trait]
impl VoiceEventHandler for VoiceTickHandler {
    async fn act(&self, ctx: &EventContext<'_>) -> Option<Event> {
        let EventContext::VoiceTick(tick) = ctx else {
            return None;
        };

        let mut currently_speaking = std::collections::HashSet::<UserId>::new();
        let ingest_params = IngestParams {
            max_ingest_samples: (self.rolling_ingest_max_ms as usize) * 16,
            keep_context_samples: (self.rolling_ingest_context_ms as usize) * 16,
            silence_ticks_threshold: self.silence_ticks_threshold,
        };

        for (ssrc, data) in &tick.speaking {
            if data.decoded_voice.is_none() && data.packet.is_some() {
                self.runtime
                    .decode_failure_activity
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }

            let Some(decoded) = &data.decoded_voice else {
                continue;
            };

            let Some(user_id) = self
                .ssrc_to_user
                .get(&(self.guild_id, *ssrc))
                .map(|v| *v)
            else {
                self.runtime
                    .unmapped_ssrc_activity
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                push_unknown_ssrc_audio(self.guild_id, *ssrc, decoded);
                continue;
            };

            self.runtime
                .decoded_audio_activity
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            let user_key = (self.guild_id, user_id);

            let mut stream = self.streams.entry(user_key).or_default();
            let mut replay = take_unknown_ssrc_audio(self.guild_id, *ssrc);
            let processed = if replay.is_empty() {
                stream.frontend.push_stereo_pcm(decoded, self.enable_denoiser)
            } else {
                replay.extend_from_slice(decoded);
                stream.frontend.push_stereo_pcm(&replay, self.enable_denoiser)
            };
            let resample_errors = stream.frontend.take_resample_error_count();
            if resample_errors > 0 {
                self.runtime
                    .resample_error_total
                    .fetch_add(resample_errors, std::sync::atomic::Ordering::SeqCst);
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
                    .compare_exchange(false, true, std::sync::atomic::Ordering::SeqCst, std::sync::atomic::Ordering::SeqCst)
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
            let cleaned_len = cleaned.len();
            let noise_rms_ema = stream.frontend.noise_rms_ema();
            let maybe_rollover_final = match append_processed_chunk(
                &mut stream.buffer,
                ProcessedSpeechChunk {
                    pcm_16k: cleaned,
                    speech_active: processed.speech_active,
                },
                ingest_params,
                Instant::now(),
            ) {
                IngestOutcome::Rollover { start_ts, pcm, voiced_ticks, .. } => {
                    Some((start_ts, pcm, voiced_ticks, noise_rms_ema))
                }
                _ => None,
            };

            tracing::trace!(
                guild = %self.guild_id,
                user = %user_id,
                ssrc = %ssrc,
                samples = cleaned_len,
                "processed 16k pcm tick"
            );

            if let Some((start_ts, pcm, voiced_ticks, noise_rms_ema)) = maybe_rollover_final {
                if let Err(rejection) = should_dispatch_chunk(&pcm, voiced_ticks, noise_rms_ema) {
                    self.runtime
                        .dispatch_gate_total
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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

        let tracked_users: Vec<(GuildId, UserId)> = self.streams.iter().map(|e| *e.key()).collect();
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
                if let IngestOutcome::Endpoint {
                    start_ts,
                    mut pcm,
                    voiced_ticks,
                    silent_ticks,
                } = advance_silence(&mut stream.buffer, ingest_params, Instant::now())
                {
                    trim_finalize_tail(&mut pcm, silent_ticks);
                    let noise_rms_ema = stream.frontend.noise_rms_ema();
                    maybe_job = Some((start_ts, pcm, voiced_ticks, noise_rms_ema));
                }
            }

            if let Some((start_ts, pcm, voiced_ticks, noise_rms_ema)) = maybe_job {
                if let Err(rejection) = should_dispatch_chunk(&pcm, voiced_ticks, noise_rms_ema) {
                    self.runtime
                        .dispatch_gate_total
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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

fn append_processed_chunk(
    buffer: &mut UserAudioBuffer,
    chunk: ProcessedSpeechChunk,
    params: IngestParams,
    now: Instant,
) -> IngestOutcome {
    if chunk.pcm_16k.is_empty() || (buffer.utterance_start.is_none() && !chunk.speech_active) {
        return IngestOutcome::None;
    }

    if buffer.utterance_start.is_none() {
        buffer.utterance_start = Some(now);
    }
    if chunk.speech_active {
        buffer.silent_ticks = 0;
        buffer.voiced_ticks = buffer.voiced_ticks.saturating_add(1);
    }
    buffer.pcm.extend(chunk.pcm_16k);

    if buffer.pcm.len() < params.max_ingest_samples {
        return IngestOutcome::None;
    }

    let max_keep_without_starving = params.max_ingest_samples.saturating_sub(1_600);
    let keep = params
        .keep_context_samples
        .min(max_keep_without_starving)
        .min(buffer.pcm.len());
    let split_at = choose_rollover_split_index(&buffer.pcm, keep, max_keep_without_starving);
    if split_at < 1_600 {
        return IngestOutcome::None;
    }

    let start_ts = buffer.utterance_start.take().unwrap_or(now);
    let source_len = buffer.pcm.len();
    let tail = buffer.pcm.split_off(split_at);
    let pcm = std::mem::replace(&mut buffer.pcm, tail);
    let carried_tail_ms = (buffer.pcm.len() as u64) / 16;
    let voiced_ticks = ((buffer.voiced_ticks as u64 * split_at as u64) / source_len as u64) as u32;
    buffer.voiced_ticks = buffer.voiced_ticks.saturating_sub(voiced_ticks);
    buffer.utterance_start = Some(
        now.checked_sub(Duration::from_millis(carried_tail_ms))
            .unwrap_or(now),
    );

    IngestOutcome::Rollover {
        start_ts,
        pcm,
        voiced_ticks,
    }
}

fn advance_silence(
    buffer: &mut UserAudioBuffer,
    params: IngestParams,
    now: Instant,
) -> IngestOutcome {
    buffer.silent_ticks = buffer.silent_ticks.saturating_add(1);
    if buffer.silent_ticks < params.silence_ticks_threshold || buffer.pcm.is_empty() {
        return IngestOutcome::None;
    }

    let start_ts = buffer.utterance_start.take().unwrap_or(now);
    let pcm = std::mem::take(&mut buffer.pcm);
    let voiced_ticks = std::mem::take(&mut buffer.voiced_ticks);
    let silent_ticks = std::mem::take(&mut buffer.silent_ticks);
    IngestOutcome::Endpoint {
        start_ts,
        pcm,
        voiced_ticks,
        silent_ticks,
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
        return max_split;
    }

    let nominal = len.saturating_sub(keep_samples).clamp(min_split, max_split);
    let search_radius = (keep_samples / 2).clamp(800, 8_000);
    let search_start = nominal.saturating_sub(search_radius).max(min_split);
    let search_end = nominal.saturating_add(search_radius).min(max_split);

    let frame = 800usize;
    let hop = 160usize;
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

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{
        IngestOutcome, IngestParams, advance_silence, append_processed_chunk,
        choose_rollover_split_index, ProcessedSpeechChunk, UserAudioBuffer,
    };

    fn params() -> IngestParams {
        IngestParams {
            max_ingest_samples: 3_200,
            keep_context_samples: 1_600,
            silence_ticks_threshold: 3,
        }
    }

    fn chunk(samples: usize, speech_active: bool) -> ProcessedSpeechChunk {
        ProcessedSpeechChunk {
            pcm_16k: vec![0.1; samples],
            speech_active,
        }
    }

    #[test]
    fn rollover_short_buffer_uses_nominal_tail_keep() {
        let pcm = vec![0.1; 2_000];
        let split = choose_rollover_split_index(&pcm, 400, 1_000);
        assert_eq!(split, 1_600);
    }

    #[test]
    fn rollover_degenerate_bounds_returns_max_split() {
        let pcm = vec![0.1; 3_200];
        let split = choose_rollover_split_index(&pcm, 800, 1_000);
        assert_eq!(split, 1_600);
    }

    #[test]
    fn rollover_split_stays_inside_bounds() {
        let pcm = vec![0.1; 10_000];
        let keep = 3_000;
        let max_keep_without_starving = 5_000;
        let split = choose_rollover_split_index(&pcm, keep, max_keep_without_starving);

        let min_split = pcm.len().saturating_sub(max_keep_without_starving).max(1_600);
        let max_split = pcm.len().saturating_sub(1_600);
        assert!(split >= min_split && split <= max_split);
    }

    #[test]
    fn rollover_prefers_quieter_window_when_available() {
        let mut pcm = vec![1.0; 10_000];
        for sample in pcm.iter_mut().take(6_400).skip(5_600) {
            *sample = 0.0;
        }

        let split = choose_rollover_split_index(&pcm, 3_000, 5_000);
        assert!(split >= 5_400 && split <= 6_600);
    }

    #[test]
    fn append_preserves_intra_utterance_silence_after_speech_starts() {
        let mut buffer = UserAudioBuffer::default();
        let now = Instant::now();

        assert!(matches!(
            append_processed_chunk(&mut buffer, chunk(320, false), params(), now),
            IngestOutcome::None
        ));
        append_processed_chunk(&mut buffer, chunk(320, true), params(), now);
        append_processed_chunk(&mut buffer, chunk(320, false), params(), now);
        append_processed_chunk(&mut buffer, chunk(320, true), params(), now);

        assert_eq!(buffer.pcm.len(), 960);
        assert_eq!(buffer.voiced_ticks, 2);
    }

    #[test]
    fn rollover_reports_actual_carried_tail_duration_and_prorates_ticks() {
        let mut buffer = UserAudioBuffer {
            pcm: vec![0.1; 2_880],
            silent_ticks: 0,
            voiced_ticks: 9,
            utterance_start: Some(Instant::now()),
        };
        let now = Instant::now() + Duration::from_secs(1);
        let outcome = append_processed_chunk(&mut buffer, chunk(320, true), params(), now);

        let IngestOutcome::Rollover {
            pcm,
            voiced_ticks,
            ..
        } = outcome else {
            panic!("expected rollover");
        };
        assert_eq!(voiced_ticks + buffer.voiced_ticks, 10);
        assert_eq!(pcm.len() + buffer.pcm.len(), 3_200);
    }

    #[test]
    fn endpoint_fires_once_at_exact_silence_threshold() {
        let mut buffer = UserAudioBuffer {
            pcm: vec![0.1; 640],
            silent_ticks: 0,
            voiced_ticks: 2,
            utterance_start: Some(Instant::now()),
        };
        let now = Instant::now();

        assert!(matches!(advance_silence(&mut buffer, params(), now), IngestOutcome::None));
        assert!(matches!(advance_silence(&mut buffer, params(), now), IngestOutcome::None));
        assert!(matches!(
            advance_silence(&mut buffer, params(), now),
            IngestOutcome::Endpoint { .. }
        ));
        assert!(matches!(advance_silence(&mut buffer, params(), now), IngestOutcome::None));
    }

    #[test]
    fn repeated_rollovers_conserve_all_pcm_samples() {
        let mut buffer = UserAudioBuffer::default();
        let mut emitted_samples = 0usize;
        let now = Instant::now();

        for tick in 0..300 {
            let outcome = append_processed_chunk(
                &mut buffer,
                chunk(320, true),
                params(),
                now + Duration::from_millis(tick * 20),
            );
            if let IngestOutcome::Rollover { pcm, .. } = outcome {
                emitted_samples += pcm.len();
            }
        }

        assert_eq!(emitted_samples + buffer.pcm.len(), 300 * 320);
    }

    #[test]
    fn rollover_split_is_panic_free_across_degenerate_bounds() {
        for len in [3_200, 10_000, 64_000, 192_000] {
            for keep in [0, 800, 24_000, len + 1] {
                for max_keep in [0, 1_600, 62_400] {
                    let pcm = vec![0.1; len];
                    let split = choose_rollover_split_index(&pcm, keep, max_keep);
                    assert!((1_600..=len - 1_600).contains(&split));
                    assert!(len - split >= 1_600);
                }
            }
        }
    }
}
