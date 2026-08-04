use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use serenity::all::{GuildId, UserId};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig};

use super::denoiser::{compute_rms, UserDenoiseState};
use super::models::{configure_model, resolve_model_dir};

const VAD_HANGOVER_MS: u32 = 256;
const DISPATCH_GATE_MIN_VOICED_TICKS: u32 = 8;
const FINALIZE_TAIL_KEEP_MS: u32 = 200;

pub struct DispatchGateRejection {
    pub reason: &'static str,
    pub voiced_ticks: u32,
    pub rms: f32,
    pub floor: f32,
}

pub fn should_dispatch_chunk(
    pcm: &[f32],
    voiced_ticks: u32,
    noise_rms_ema: f32,
) -> Result<(), DispatchGateRejection> {
    let rms = compute_rms(pcm);
    let floor = (noise_rms_ema * 1.4).max(0.0025);

    if voiced_ticks < DISPATCH_GATE_MIN_VOICED_TICKS {
        return Err(DispatchGateRejection {
            reason: "insufficient_voiced_ticks",
            voiced_ticks,
            rms,
            floor,
        });
    }

    if rms < floor {
        return Err(DispatchGateRejection {
            reason: "below_noise_floor",
            voiced_ticks,
            rms,
            floor,
        });
    }

    Ok(())
}

pub fn trim_finalize_tail(pcm: &mut Vec<f32>, silence_ticks: u32) {
    if silence_ticks == 0 {
        return;
    }

    let estimated_tail_ms = silence_ticks
        .saturating_mul(20)
        .saturating_add(VAD_HANGOVER_MS);
    let trim_ms = estimated_tail_ms.saturating_sub(FINALIZE_TAIL_KEEP_MS);
    let trim_samples = trim_ms as usize * 16;
    let min_samples_to_keep: usize = 1_600;

    if trim_samples == 0 || pcm.len() <= min_samples_to_keep.saturating_add(trim_samples) {
        return;
    }

    pcm.truncate(pcm.len().saturating_sub(trim_samples));
}

pub struct AsrEngine {
    recognizer: Arc<OfflineRecognizer>,
}

impl AsrEngine {
    pub fn new(
        model_dir: &str,
        asr_num_threads: i32,
        forced_family_hint: Option<&str>,
    ) -> anyhow::Result<Self> {
        let model_base = resolve_model_dir(model_dir)?;
        let mut cfg = OfflineRecognizerConfig::default();

        let selected_backend = configure_model(&mut cfg, &model_base, forced_family_hint)?;

        cfg.model_config.num_threads = asr_num_threads.clamp(1, 8);

        let recognizer = OfflineRecognizer::create(&cfg).ok_or_else(|| {
            anyhow::anyhow!(
                "sherpa-onnx failed to create a recognizer for the {} backend from {}",
                selected_backend,
                model_base.display()
            )
        })?;

        tracing::info!(
            "ASR backend selected: {} (model_dir={}, num_threads={})",
            selected_backend,
            model_base.display(),
            cfg.model_config.num_threads
        );

        if selected_backend.contains("whisper") {
            tracing::warn!(
                "Whisper backend selected: offline decode cost is effectively fixed by padded context; short conversational turns may incur high latency/backlog."
            );
        }

        Ok(Self { recognizer: Arc::new(recognizer) })
    }

    pub fn transcribe_16k_mono(&self, samples: &[f32]) -> String {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(16_000, samples);
        self.recognizer.decode(&stream);
        stream
            .get_result()
            .map(|r| r.text)
            .unwrap_or_default()
            .trim()
            .to_string()
    }
}

#[derive(Default, Clone)]
pub struct UserAudioBuffer {
    pub pcm: Vec<f32>,
    pub silent_ticks: u32,
    pub voiced_ticks: u32,
    pub utterance_start: Option<Instant>,
}

#[derive(Default)]
pub struct UserStreamState {
    pub denoiser: UserDenoiseState,
    pub buffer: UserAudioBuffer,
}

pub async fn transcribe_mono_pcm(
    asr: Arc<AsrEngine>,
    pcm_mono: Vec<f32>,
) -> Option<String> {
    let sample_count = pcm_mono.len();
    if sample_count < 1600 {
        return None;
    }

    let decode_started = Instant::now();
    let text = tokio::task::spawn_blocking(move || asr.transcribe_16k_mono(&pcm_mono))
        .await
        .ok()?;
    let decode_ms = decode_started.elapsed().as_millis() as u64;

    tracing::debug!(samples = sample_count, decode_ms, "asr decode completed");

    let text = text.trim().to_string();
    if text.is_empty() {
        return None;
    }

    let audio_secs = (sample_count as f32 / 16_000.0).max(0.001);
    if let Some(reason) = decode_rejection_reason(&text, audio_secs) {
        tracing::warn!(reason, samples = sample_count, "rejected decoded utterance");
        return None;
    }

    Some(text)
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

    if HALLUCINATION_BLOCKLIST.iter().any(|phrase| normalized == *phrase) {
        return Some("blocklist");
    }

    let chars_per_second = text.chars().count() as f32 / audio_secs;
    if chars_per_second > 25.0 {
        return Some("implausible_char_rate");
    }

    None
}

pub type Streams = DashMap<(GuildId, UserId), UserStreamState>;
pub type SsrcMap = DashMap<(GuildId, u32), UserId>;
