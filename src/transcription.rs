use std::fs;
use std::path::{Path, PathBuf};
use std::collections::{HashMap, HashSet};
use std::io::{BufWriter, Write};
use std::thread;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use dashmap::DashMap;
use nnnoiseless::DenoiseState;
use rubato::{FftFixedInOut, Resampler};
use serenity::all::{GuildId, UserId};
use sherpa_onnx::{
    OfflineMoonshineModelConfig, OfflineNemoEncDecCtcModelConfig, OfflineParaformerModelConfig,
    OfflineQwen3ASRModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSenseVoiceModelConfig,
    OfflineTdnnModelConfig, OfflineTransducerModelConfig, OfflineWhisperModelConfig,
    OfflineZipformerCtcModelConfig,
    OnlineRecognizer, OnlineRecognizerConfig, OnlineTransducerModelConfig,
};
use tokio::sync::{mpsc, RwLock};
use tokio::sync::Semaphore;
use serde::Serialize;

use crate::app::{CallSession, LivePartialSnapshot, Utterance, UtteranceStage};

pub const REORDER_WINDOW: Duration = Duration::from_millis(200);
pub const RNNOISE_FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;
const HIGHPASS_ALPHA_48K: f32 = 0.991;

pub struct AsrEngine {
    recognizer: Arc<OfflineRecognizer>,
}

pub struct OnlineAsrEngine {
    #[allow(dead_code)]
    recognizer: Arc<OnlineRecognizer>,
}

pub enum StreamingDecoderCommand {
    AudioChunk {
        user_id: UserId,
        pcm_16k: Vec<f32>,
        observed_at: Instant,
    },
    TickDone {
        heard_users: Vec<UserId>,
        observed_at: Instant,
    },
    FlushAll {
        respond_to: tokio::sync::oneshot::Sender<()>,
        observed_at: Instant,
    },
}

struct StreamingStreamState {
    stream: sherpa_onnx::OnlineStream,
    utterance_seq: u64,
    stream_anchor_at: Option<Instant>,
    total_samples_fed: u64,
    utterance_start_sample: Option<u64>,
    last_partial_text: String,
    last_emit_at: Option<Instant>,
    dormant_after_endpoint: bool,
    dormant_silence_ticks: u32,
    pcm_16k: Vec<f32>,
    last_clock_drift_log_at: Option<Instant>,
}

impl StreamingStreamState {
    fn new(stream: sherpa_onnx::OnlineStream, utterance_seq: u64) -> Self {
        Self {
            stream,
            utterance_seq,
            stream_anchor_at: None,
            total_samples_fed: 0,
            utterance_start_sample: None,
            last_partial_text: String::new(),
            last_emit_at: None,
            dormant_after_endpoint: false,
            dormant_silence_ticks: 0,
            pcm_16k: Vec::new(),
            last_clock_drift_log_at: None,
        }
    }
}

pub struct OfflineFinalizeJob {
    pub user_id: UserId,
    pub start_ts: Instant,
    pub start_offset_ms: u64,
    pub revision_id: u64,
    pub stream_final_text: String,
    pub pcm_16k: Vec<f32>,
}

struct DecodeTextResult {
    text: String,
    tokens: Vec<String>,
    token_timestamps_s: Vec<f32>,
    decode_elapsed_ms: u64,
}

/// Explicit override for the single-file model families that are indistinguishable
/// from filenames alone (Paraformer / SenseVoice / NeMo CTC / Zipformer CTC / TDNN
/// are all typically just "model.onnx" + "tokens.txt"). Set via ASR_MODEL_FAMILY.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForcedFamily {
    Paraformer,
    SenseVoice,
    NemoCtc,
    ZipformerCtc,
    Tdnn,
}

impl ForcedFamily {
    fn from_env() -> Option<Self> {
        let raw = std::env::var("ASR_MODEL_FAMILY").ok()?;
        match raw.to_lowercase().replace('-', "_").as_str() {
            "paraformer" => Some(Self::Paraformer),
            "sense_voice" | "sensevoice" => Some(Self::SenseVoice),
            "nemo_ctc" | "nemoctc" => Some(Self::NemoCtc),
            "zipformer_ctc" | "zipformerctc" => Some(Self::ZipformerCtc),
            "tdnn" => Some(Self::Tdnn),
            _ => None,
        }
    }

    /// Soft guess from the model directory's own name, matching the conventional
    /// naming sherpa-onnx's published bundles use (e.g. "sherpa-onnx-sense-voice-...",
    /// "sherpa-onnx-zipformer-ctc-...", "sherpa-onnx-nemo-...").
    fn guess_from_dir_name(dir: &Path) -> Option<Self> {
        let name = dir.file_name()?.to_str()?.to_lowercase();
        if name.contains("sense-voice") || name.contains("sense_voice") || name.contains("sensevoice") {
            Some(Self::SenseVoice)
        } else if name.contains("paraformer") {
            Some(Self::Paraformer)
        } else if name.contains("zipformer") && name.contains("ctc") {
            Some(Self::ZipformerCtc)
        } else if name.contains("nemo") || name.contains("giga-am") || name.contains("gigaam") {
            Some(Self::NemoCtc)
        } else if name.contains("tdnn") {
            Some(Self::Tdnn)
        } else {
            None
        }
    }
}

impl AsrEngine {
    #[allow(dead_code)]
    pub fn new(model_dir: &str) -> anyhow::Result<Self> {
        Self::new_with_threads(model_dir, None)
    }

    pub fn new_with_threads(model_dir: &str, num_threads: Option<i32>) -> anyhow::Result<Self> {
        let model_base = resolve_model_dir(model_dir)?;
        let mut cfg = OfflineRecognizerConfig::default();

        let selected_backend: &str = if let Some(label) = try_transducer(&mut cfg, &model_base) {
            label
        } else if let Some(label) = try_moonshine(&mut cfg, &model_base) {
            label
        } else if let Some(label) = try_whisper(&mut cfg, &model_base)? {
            label
        } else if let Some(label) = try_qwen3_asr(&mut cfg, &model_base) {
            label
        } else if let Some(label) = try_single_file_family(&mut cfg, &model_base)? {
            label
        } else {
            anyhow::bail!(
                "could not identify a supported ASR model family in {} \
                 (looked for transducer encoder/decoder/joiner, Whisper encoder/decoder, \
                 Moonshine's split or merged files, and single-file model.onnx variants)",
                model_base.display()
            );
        };

        cfg.model_config.num_threads = num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4)
                .clamp(1, 8)
        });

        let recognizer = OfflineRecognizer::create(&cfg).ok_or_else(|| {
            anyhow::anyhow!(
                "sherpa-onnx failed to create a recognizer for the {} backend from {}",
                selected_backend,
                model_base.display()
            )
        })?;

        tracing::info!(
            "Offline ASR backend selected: {} ({})",
            selected_backend,
            model_base.display()
        );

        Ok(Self { recognizer: Arc::new(recognizer) })
    }

    fn transcribe_16k_mono(&self, samples: &[f32]) -> DecodeTextResult {
        let started = Instant::now();
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(16_000, samples);
        self.recognizer.decode(&stream);
        let result = stream.get_result();
        let text = result
            .as_ref()
            .map(|r| r.text.trim().to_string())
            .unwrap_or_default();
        let tokens = result
            .as_ref()
            .map(|r| r.tokens.clone())
            .unwrap_or_default();
        let token_timestamps_s = result
            .as_ref()
            .map(|r| r.timestamps.clone())
            .flatten()
            .unwrap_or_default();
        DecodeTextResult {
            text,
            tokens,
            token_timestamps_s,
            decode_elapsed_ms: started.elapsed().as_millis() as u64,
        }
    }

}

impl OnlineAsrEngine {
    pub fn new(model_dir: &str) -> anyhow::Result<Self> {
        let model_base = resolve_model_dir(model_dir)?;
        let mut cfg = OnlineRecognizerConfig::default();

        let encoder = find_by_prefix(&model_base, "encoder").ok_or_else(|| {
            anyhow::anyhow!(
                "could not find streaming encoder*.onnx in {}",
                model_base.display()
            )
        })?;
        let decoder = find_by_prefix(&model_base, "decoder").ok_or_else(|| {
            anyhow::anyhow!(
                "could not find streaming decoder*.onnx in {}",
                model_base.display()
            )
        })?;
        let joiner = find_by_prefix(&model_base, "joiner").ok_or_else(|| {
            anyhow::anyhow!(
                "could not find streaming joiner*.onnx in {}",
                model_base.display()
            )
        })?;
        let tokens = find_tokens_file(&model_base).ok_or_else(|| {
            anyhow::anyhow!(
                "could not find streaming tokens.txt in {}",
                model_base.display()
            )
        })?;

        cfg.model_config.transducer = OnlineTransducerModelConfig {
            encoder: Some(encoder.to_string_lossy().to_string()),
            decoder: Some(decoder.to_string_lossy().to_string()),
            joiner: Some(joiner.to_string_lossy().to_string()),
        };
        cfg.model_config.tokens = Some(tokens.to_string_lossy().to_string());
        cfg.model_config.num_threads = 1;
        cfg.enable_endpoint = true;
        cfg.rule1_min_trailing_silence = 2.4;
        cfg.rule2_min_trailing_silence = 1.5;
        cfg.rule3_min_utterance_length = 20.0;
        cfg.decoding_method = Some("greedy_search".to_string());

        let recognizer = OnlineRecognizer::create(&cfg).ok_or_else(|| {
            anyhow::anyhow!(
                "sherpa-onnx failed to create an online recognizer from {}",
                model_base.display()
            )
        })?;

        tracing::info!(
            "Streaming ASR backend selected: transducer (ASR_STREAMING_MODEL_DIR={})",
            model_base.display()
        );

        Ok(Self {
            recognizer: Arc::new(recognizer),
        })
    }

    #[allow(dead_code)]
    pub fn recognizer(&self) -> &Arc<OnlineRecognizer> {
        &self.recognizer
    }
}

/// Finds the first file in `dir` whose name starts with `prefix` and ends with `.onnx`,
/// preferring int8-quantized variants when more than one candidate exists (smaller and
/// faster on a Pi 5). This handles both fixed names ("encoder.onnx") and the
/// epoch-numbered names Zipformer/icefall bundles ship ("encoder-epoch-99-avg-1.int8.onnx").
fn find_by_prefix(dir: &Path, prefix: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with(prefix) && n.ends_with(".onnx"))
                    .unwrap_or(false)
        })
        .collect();

    candidates.sort_by_key(|p| {
        let is_int8 = p
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.contains("int8"))
            .unwrap_or(false);
        (!is_int8, p.clone()) // int8 variants sort first
    });

    candidates.into_iter().next()
}

/// Find token file, supporting both `tokens.txt` and prefixed variants like
/// `base.en-tokens.txt`.
fn find_tokens_file(dir: &Path) -> Option<PathBuf> {
    let exact = dir.join("tokens.txt");
    if exact.is_file() {
        return Some(exact);
    }

    let mut candidates: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| {
                        n.ends_with(".txt")
                            && (n == "tokens.txt"
                                || n.contains("-tokens")
                                || n.contains("_tokens")
                                || n.contains(".tokens"))
                    })
                    .unwrap_or(false)
        })
        .collect();

    candidates.sort();
    candidates.into_iter().next()
}

/// Like `find_by_prefix`, but also accepts names where `hint` appears after a delimiter,
/// e.g. `base.en-encoder.int8.onnx` or `model_decoder.onnx`.
fn find_by_hint(dir: &Path, hint: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| {
                        n.ends_with(".onnx")
                            && (n.starts_with(hint)
                                || n.contains(&format!("-{hint}"))
                                || n.contains(&format!("_{hint}")))
                    })
                    .unwrap_or(false)
        })
        .collect();

    candidates.sort_by_key(|p| {
        let is_int8 = p
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.contains("int8"))
            .unwrap_or(false);
        (!is_int8, p.clone())
    });

    candidates.into_iter().next()
}

fn try_transducer(cfg: &mut OfflineRecognizerConfig, dir: &Path) -> Option<&'static str> {
    let encoder = find_by_prefix(dir, "encoder")?;
    let decoder = find_by_prefix(dir, "decoder")?;
    let joiner = find_by_prefix(dir, "joiner")?;
    let tokens = find_tokens_file(dir)?;

    cfg.model_config.transducer = OfflineTransducerModelConfig {
        encoder: Some(encoder.to_string_lossy().to_string()),
        decoder: Some(decoder.to_string_lossy().to_string()),
        joiner: Some(joiner.to_string_lossy().to_string()),
    };
    cfg.model_config.tokens = Some(tokens.to_string_lossy().to_string());
    Some("transducer (Zipformer / NeMo Parakeet-style)")
}

fn try_moonshine(cfg: &mut OfflineRecognizerConfig, dir: &Path) -> Option<&'static str> {
    let preprocess = dir.join("preprocess.onnx");
    let encode = find_by_prefix(dir, "encode")?;
    let uncached = find_by_prefix(dir, "uncached_decode");
    let cached = find_by_prefix(dir, "cached_decode");
    let merged = find_by_prefix(dir, "merged_decod"); // matches merged_decoder / merged_decode
    let tokens = find_tokens_file(dir)?;

    if let (true, Some(uncached), Some(cached)) = (preprocess.is_file(), uncached, cached) {
        cfg.model_config.moonshine = OfflineMoonshineModelConfig {
            preprocessor: Some(preprocess.to_string_lossy().to_string()),
            encoder: Some(encode.to_string_lossy().to_string()),
            uncached_decoder: Some(uncached.to_string_lossy().to_string()),
            cached_decoder: Some(cached.to_string_lossy().to_string()),
            merged_decoder: None,
        };
        cfg.model_config.tokens = Some(tokens.to_string_lossy().to_string());
        Some("moonshine (split)")
    } else if let Some(merged) = merged {
        cfg.model_config.moonshine = OfflineMoonshineModelConfig {
            preprocessor: None,
            encoder: Some(encode.to_string_lossy().to_string()),
            uncached_decoder: None,
            cached_decoder: None,
            merged_decoder: Some(merged.to_string_lossy().to_string()),
        };
        cfg.model_config.tokens = Some(tokens.to_string_lossy().to_string());
        Some("moonshine (merged)")
    } else {
        None
    }
}

fn try_whisper(cfg: &mut OfflineRecognizerConfig, dir: &Path) -> anyhow::Result<Option<&'static str>> {
    let Some(encoder) = find_by_hint(dir, "encoder").or_else(|| find_by_hint(dir, "large"))
    else {
        return Ok(None);
    };
    // Whisper's own encoder/decoder pair shares the "encoder"/"decoder" prefix with
    // transducer models, but transducer also requires a joiner -- try_transducer runs
    // first, so reaching here means no joiner was found, i.e. this really is Whisper.
    let Some(decoder) = find_by_hint(dir, "decoder").or_else(|| find_by_hint(dir, "large"))
    else {
        return Ok(None);
    };
    let Some(tokens) = find_tokens_file(dir) else {
        return Ok(None);
    };

    cfg.model_config.whisper = OfflineWhisperModelConfig {
        encoder: Some(encoder.to_string_lossy().to_string()),
        decoder: Some(decoder.to_string_lossy().to_string()),
        language: Some("en".to_string()),
        task: Some("transcribe".to_string()),
        tail_paddings: -1,
        enable_token_timestamps: false,
        enable_segment_timestamps: false,
    };
    cfg.model_config.tokens = Some(tokens.to_string_lossy().to_string());
    Ok(Some("whisper"))
}

fn try_qwen3_asr(cfg: &mut OfflineRecognizerConfig, dir: &Path) -> Option<&'static str> {
    let conv_frontend = dir.join("conv_frontend.onnx");
    if !conv_frontend.is_file() {
        return None;
    }

    let encoder = find_by_hint(dir, "encoder")?;
    let decoder = find_by_hint(dir, "decoder")?;
    let tokenizer = dir.join("tokenizer");
    if !tokenizer.is_dir() {
        return None;
    }

    cfg.model_config.qwen3_asr = OfflineQwen3ASRModelConfig {
        conv_frontend: Some(conv_frontend.to_string_lossy().to_string()),
        encoder: Some(encoder.to_string_lossy().to_string()),
        decoder: Some(decoder.to_string_lossy().to_string()),
        tokenizer: Some(tokenizer.to_string_lossy().to_string()),
        ..Default::default()
    };

    Some("qwen3_asr")
}

fn try_single_file_family(
    cfg: &mut OfflineRecognizerConfig,
    dir: &Path,
) -> anyhow::Result<Option<&'static str>> {
    let model = ["model.onnx", "model.int8.onnx"]
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.is_file());
    let Some(model) = model else { return Ok(None) };

    let tokens = dir.join("tokens.txt");
    if !tokens.is_file() {
        return Ok(None);
    }

    let family = ForcedFamily::from_env()
        .or_else(|| ForcedFamily::guess_from_dir_name(dir))
        .ok_or_else(|| anyhow::anyhow!(
            "found a single model.onnx in {} but can't tell which family it is -- \
             Paraformer, SenseVoice, NeMo CTC, Zipformer CTC, and TDNN models are all \
             shipped this way and are not distinguishable by filename alone. \
             Set ASR_MODEL_FAMILY to one of: paraformer, sense_voice, nemo_ctc, zipformer_ctc, tdnn",
            dir.display()
        ))?;

    let model_str = Some(model.to_string_lossy().to_string());
    let tokens_str = Some(tokens.to_string_lossy().to_string());

    let label = match family {
        ForcedFamily::Paraformer => {
            cfg.model_config.paraformer = OfflineParaformerModelConfig { model: model_str };
            "paraformer"
        }
        ForcedFamily::SenseVoice => {
            cfg.model_config.sense_voice = OfflineSenseVoiceModelConfig {
                model: model_str,
                language: Some("auto".to_string()),
                use_itn: true,
            };
            "sense_voice"
        }
        ForcedFamily::NemoCtc => {
            cfg.model_config.nemo_ctc = OfflineNemoEncDecCtcModelConfig { model: model_str };
            "nemo_ctc"
        }
        // NOTE: verify these two structs' exact field name against the sherpa-onnx docs
        // for your pinned crate version before relying on them -- unlike paraformer/
        // sense_voice/nemo_ctc, I haven't directly confirmed OfflineZipformerCtcModelConfig
        // and OfflineTdnnModelConfig's field name (assumed `model` by pattern, matching
        // every other single-file config in this crate).
        ForcedFamily::ZipformerCtc => {
            cfg.model_config.zipformer_ctc = OfflineZipformerCtcModelConfig { model: model_str };
            "zipformer_ctc"
        }
        ForcedFamily::Tdnn => {
            cfg.model_config.tdnn = OfflineTdnnModelConfig { model: model_str };
            "tdnn"
        }
    };

    cfg.model_config.tokens = tokens_str;
    Ok(Some(label))
}

fn resolve_model_dir(model_dir: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(model_dir);
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }

    let cwd = std::env::current_dir().context("failed to read current working directory")?;
    Ok(cwd.join(path))
}

pub struct UserDenoiseState {
    denoiser: Box<DenoiseState<'static>>,
    pending: Vec<f32>,
    warmed_up: bool,
    hp_prev_x: f32,
    hp_prev_y: f32,
    resampler_48k_to_16k: FftFixedInOut<f32>,
    resample_pending_48k: Vec<f32>,
}

impl UserDenoiseState {
    pub fn new() -> Self {
        Self {
            denoiser: DenoiseState::new(),
            pending: Vec::new(),
            warmed_up: false,
            hp_prev_x: 0.0,
            hp_prev_y: 0.0,
            resampler_48k_to_16k: FftFixedInOut::new(48_000, 16_000, 960, 1)
                .expect("valid fixed 48k->16k resampler config"),
            resample_pending_48k: Vec::new(),
        }
    }

    pub fn push_stereo_pcm_hybrid(&mut self, input: &[i16], enable_denoiser: bool) -> Vec<f32> {
        let mut mono = downmix_stereo_to_mono_unit_scale(input);
        if mono.is_empty() {
            return Vec::new();
        }

        self.apply_highpass(&mut mono);

        let cleaned_48k = if enable_denoiser {
            self.push_mono_pcm(&mono)
        } else {
            let mut passthrough = self.drain_pending_passthrough();
            passthrough.extend_from_slice(&mono);
            passthrough
        };

        self.resample_48k_to_16k_stream(&cleaned_48k)
    }

    pub fn push_mono_pcm(&mut self, input: &[f32]) -> Vec<f32> {
        self.pending.extend_from_slice(input);
        let mut out = Vec::new();

        while self.pending.len() >= RNNOISE_FRAME_SIZE {
            let frame: Vec<f32> = self.pending.drain(..RNNOISE_FRAME_SIZE).collect();
            out.extend(self.denoise_frame(&frame));
        }

        out
    }

    fn drain_pending_passthrough(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.pending)
    }

    fn apply_highpass(&mut self, samples: &mut [f32]) {
        for sample in samples {
            let x = *sample;
            let y = HIGHPASS_ALPHA_48K * (self.hp_prev_y + x - self.hp_prev_x);
            self.hp_prev_x = x;
            self.hp_prev_y = y;
            *sample = y;
        }
    }

    fn resample_48k_to_16k_stream(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        self.resample_pending_48k.extend_from_slice(input);

        let in_frames = self.resampler_48k_to_16k.input_frames_next();
        let out_frames = self.resampler_48k_to_16k.output_frames_next();

        let mut out = Vec::with_capacity(
            (self.resample_pending_48k.len() / in_frames)
                .saturating_mul(out_frames),
        );

        while self.resample_pending_48k.len() >= in_frames {
            let in_chunk: Vec<f32> = self.resample_pending_48k.drain(..in_frames).collect();
            let mut out_chunk = vec![0.0f32; out_frames];
            if self
                .resampler_48k_to_16k
                .process_into_buffer(
                    &[in_chunk.as_slice()],
                    &mut [out_chunk.as_mut_slice()],
                    None,
                )
                .is_err()
            {
                tracing::warn!("48k->16k resample chunk failed; dropping chunk");
                continue;
            }
            out.extend_from_slice(&out_chunk);
        }

        out
    }

    fn denoise_frame(&mut self, input: &[f32]) -> Vec<f32> {
        let mut scaled = [0.0f32; RNNOISE_FRAME_SIZE];
        for (dst, src) in scaled.iter_mut().zip(input.iter().copied()) {
            *dst = src * i16::MAX as f32;
        }

        let mut output = [0.0f32; RNNOISE_FRAME_SIZE];
        self.denoiser.process_frame(&mut output, &scaled);

        if !self.warmed_up {
            self.warmed_up = true;
            return Vec::new();
        }

        output
            .into_iter()
            .map(|sample| sample / i16::MAX as f32)
            .collect()
    }
}

impl Default for UserDenoiseState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
pub struct UserStreamState {
    pub denoiser: UserDenoiseState,
}

pub fn make_revision_id(user_id: UserId, revision_seq: u64) -> u64 {
    user_id.get().wrapping_mul(0x9E37_79B1_85EB_CA87) ^ revision_seq
}

pub fn downmix_stereo_to_mono_i16_scale(input: &[i16]) -> Vec<f32> {
    let mut out = Vec::with_capacity(input.len() / 2);
    let mut i = 0;
    while i + 1 < input.len() {
        let l = input[i] as f32;
        let r = input[i + 1] as f32;
        out.push((l + r) * 0.5);
        i += 2;
    }
    out
}

pub fn downmix_stereo_to_mono_unit_scale(input: &[i16]) -> Vec<f32> {
    downmix_stereo_to_mono_i16_scale(input)
        .into_iter()
        .map(|v| (v / i16::MAX as f32).clamp(-1.0, 1.0))
        .collect()
}

async fn transcribe_mono_pcm(
    asr: Arc<AsrEngine>,
    pcm_mono: Vec<f32>,
) -> Option<DecodeTextResult> {
    if pcm_mono.len() < 1600 {
        return None;
    }

    // Bound concurrent ASR decode work to keep CPU usage stable on small devices.
    let permit = asr_decode_semaphore().acquire_owned().await.ok()?;

    let text = tokio::task::spawn_blocking(move || asr.transcribe_16k_mono(&pcm_mono))
        .await
        .ok()?;
    drop(permit);

    Some(text)
}

fn asr_decode_semaphore() -> Arc<Semaphore> {
    static ASR_DECODE_SEMAPHORE: OnceLock<Arc<Semaphore>> = OnceLock::new();
    Arc::clone(ASR_DECODE_SEMAPHORE.get_or_init(|| Arc::new(Semaphore::new(1))))
}

pub fn should_accept_offline_final(stream_final: &str, offline_final: &str) -> bool {
    if offline_final.trim().is_empty() {
        return false;
    }

    // Prefer offline final even when lexical content is equivalent; it carries
    // better punctuation/casing and produces a more uniform transcript.
    if texts_equivalent(stream_final, offline_final) {
        return true;
    }

    // Streaming hypotheses are the most likely place for loop artifacts.
    // If stream loops and offline does not, prefer offline before ratio checks.
    if has_repeated_ngram(stream_final, 3, 3) && !has_repeated_ngram(offline_final, 3, 3) {
        return true;
    }

    if has_repeated_ngram(offline_final, 3, 3) {
        return false;
    }

    let w1 = stream_final.split_whitespace().count().max(1);
    let w2 = offline_final.split_whitespace().count();
    let ratio = w2 as f32 / w1 as f32;
    (0.4..=3.0).contains(&ratio)
}

fn normalize_streaming_text(raw: &str) -> String {
    raw.trim().to_string()
}

fn polish_stream_final_text(raw: &str) -> String {
    let compact = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.is_empty() {
        return compact;
    }

    let mut chars = compact.chars();
    let mut out = String::with_capacity(compact.len() + 1);
    if let Some(first) = chars.next() {
        if first.is_ascii_alphabetic() {
            out.push(first.to_ascii_uppercase());
        } else {
            out.push(first);
        }
    }
    out.extend(chars);

    let has_terminal_punct = out
        .chars()
        .last()
        .map(|c| matches!(c, '.' | '!' | '?'))
        .unwrap_or(false);
    if !has_terminal_punct {
        out.push('.');
    }

    out
}

fn sample_clock_to_instant(anchor: Instant, sample_index: u64) -> Instant {
    let ms = sample_index.saturating_mul(1000) / 16_000;
    anchor + Duration::from_millis(ms)
}

fn sample_clock_to_offset_ms(sample_index: u64) -> u64 {
    sample_index.saturating_mul(1000) / 16_000
}

fn trim_trailing_silence_keep_tail(samples: &mut Vec<f32>, keep_tail_samples: usize) {
    if samples.is_empty() {
        return;
    }

    let last_non_silent = samples.iter().rposition(|x| x.abs() > 1.0e-6);
    let Some(last_non_silent) = last_non_silent else {
        samples.truncate(keep_tail_samples.min(samples.len()));
        return;
    };

    let target_len = (last_non_silent + 1).saturating_add(keep_tail_samples);
    if target_len < samples.len() {
        samples.truncate(target_len);
    }
}

pub fn streaming_decoder_loop(
    guild_id: GuildId,
    online_asr: Arc<OnlineAsrEngine>,
    mut rx: mpsc::Receiver<StreamingDecoderCommand>,
    utterance_tx: mpsc::Sender<Utterance>,
    offline_finalize_tx: mpsc::Sender<OfflineFinalizeJob>,
    offline_finalize_dropped: Arc<AtomicUsize>,
    offline_finalize_inflight: Arc<AtomicUsize>,
    pending_commits: Arc<AtomicUsize>,
    live_partial_text: Arc<DashMap<(GuildId, UserId), LivePartialSnapshot>>,
    live_transcript_debug: bool,
    decoder_thread_alive: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    const STREAMING_SILENCE_SAMPLES: usize = 320;
    const PARTIAL_EMIT_MIN_INTERVAL: Duration = Duration::from_millis(250);
    const DORMANT_TEARDOWN_TICKS: u32 = 100;
    const MAX_DECODE_BATCHES_PER_TICK: usize = 64;

    thread::Builder::new()
        .name(format!("stream-decoder-{}", guild_id.get()))
        .spawn(move || {
    struct DecoderAliveGuard {
        alive: Arc<AtomicBool>,
    }
    impl Drop for DecoderAliveGuard {
        fn drop(&mut self) {
            self.alive.store(false, AtomicOrdering::SeqCst);
        }
    }

    let _alive_guard = DecoderAliveGuard {
        alive: Arc::clone(&decoder_thread_alive),
    };

    let recognizer = online_asr.recognizer();
    let silence = [0.0f32; STREAMING_SILENCE_SAMPLES];
    let mut streams = HashMap::<UserId, StreamingStreamState>::new();
    let mut next_utterance_seq = HashMap::<UserId, u64>::new();

    while let Some(command) = rx.blocking_recv() {
        match command {
            StreamingDecoderCommand::AudioChunk {
                user_id,
                pcm_16k,
                observed_at,
            } => {
                let state = streams
                    .entry(user_id)
                    .or_insert_with(|| {
                        StreamingStreamState::new(
                            recognizer.create_stream(),
                            *next_utterance_seq.get(&user_id).unwrap_or(&0),
                        )
                    });
                if state.dormant_after_endpoint {
                    state.pcm_16k.clear();
                }
                if state.stream_anchor_at.is_none() {
                    state.stream_anchor_at = Some(observed_at);
                }
                if state.utterance_start_sample.is_none() {
                    state.utterance_start_sample = Some(state.total_samples_fed);
                }
                state.dormant_after_endpoint = false;
                state.dormant_silence_ticks = 0;
                state.pcm_16k.extend_from_slice(&pcm_16k);
                state.stream.accept_waveform(16_000, &pcm_16k);
                state.total_samples_fed = state
                    .total_samples_fed
                    .saturating_add(pcm_16k.len() as u64);
            }
            StreamingDecoderCommand::TickDone {
                heard_users,
                observed_at,
            } => {
                let heard = heard_users.into_iter().collect::<HashSet<_>>();
                let tracked_users = streams.keys().copied().collect::<Vec<_>>();
                let mut to_drop = Vec::new();

                for user_id in tracked_users {
                    let Some(state) = streams.get_mut(&user_id) else {
                        continue;
                    };

                    if heard.contains(&user_id) {
                        state.dormant_silence_ticks = 0;
                        if let Some(anchor) = state.stream_anchor_at {
                            let sample_elapsed_ms = state.total_samples_fed.saturating_mul(1000) / 16_000;
                            let wall_elapsed_ms = observed_at
                                .saturating_duration_since(anchor)
                                .as_millis() as u64;
                            let drift_ms = sample_elapsed_ms.abs_diff(wall_elapsed_ms);
                            let should_log = drift_ms > 200
                                && state
                                    .last_clock_drift_log_at
                                    .map(|t| observed_at.saturating_duration_since(t) >= Duration::from_secs(10))
                                    .unwrap_or(true);
                            if should_log {
                                state.last_clock_drift_log_at = Some(observed_at);
                                tracing::warn!(
                                    guild = %guild_id,
                                    user = %user_id,
                                    sample_elapsed_ms,
                                    wall_elapsed_ms,
                                    drift_ms,
                                    "stream sample clock drift exceeds threshold"
                                );
                            }
                        }
                        continue;
                    }

                    state.stream.accept_waveform(16_000, &silence);
                    state.pcm_16k.extend_from_slice(&silence);
                    state.total_samples_fed = state
                        .total_samples_fed
                        .saturating_add(silence.len() as u64);
                    if state.dormant_after_endpoint {
                        state.dormant_silence_ticks = state.dormant_silence_ticks.saturating_add(1);
                        if state.dormant_silence_ticks >= DORMANT_TEARDOWN_TICKS {
                            to_drop.push(user_id);
                        }
                    }
                }

                let mut decode_iterations = 0usize;
                loop {
                    let ready_users = streams
                        .iter()
                        .filter_map(|(user_id, state)| recognizer.is_ready(&state.stream).then_some(*user_id))
                        .collect::<Vec<_>>();
                    if ready_users.is_empty() {
                        break;
                    }
                    decode_iterations += 1;
                    if decode_iterations > MAX_DECODE_BATCHES_PER_TICK {
                        tracing::warn!(guild = %guild_id, "streaming decode loop hit iteration cap during TickDone");
                        break;
                    }

                    let ready_streams = ready_users
                        .iter()
                        .filter_map(|user_id| streams.get(user_id).map(|state| &state.stream))
                        .collect::<Vec<_>>();
                    recognizer.decode_multiple_streams(&ready_streams);

                    let mut emissions = Vec::<Utterance>::new();

                    for user_id in ready_users {
                        let Some(state) = streams.get_mut(&user_id) else {
                            continue;
                        };

                        let Some(result) = recognizer.get_result(&state.stream) else {
                            continue;
                        };

                        let text = normalize_streaming_text(&result.text);
                        let tokens = result.tokens.clone();
                        let token_timestamps_s = result.timestamps.clone().unwrap_or_default();

                        let revision_id = make_revision_id(user_id, state.utterance_seq);
                        let start_sample = state
                            .utterance_start_sample
                            .unwrap_or(state.total_samples_fed);
                        let start_ts = state
                            .stream_anchor_at
                            .map(|anchor| sample_clock_to_instant(anchor, start_sample))
                            .unwrap_or(observed_at);
                        let start_offset_ms = sample_clock_to_offset_ms(start_sample);

                        if !text.is_empty()
                            && text != state.last_partial_text
                            && state
                                .last_emit_at
                                .map(|t| observed_at.saturating_duration_since(t) >= PARTIAL_EMIT_MIN_INTERVAL)
                                .unwrap_or(true)
                        {
                            state.last_partial_text = text.clone();
                            state.last_emit_at = Some(observed_at);
                            live_partial_text.insert(
                                (guild_id, user_id),
                                LivePartialSnapshot {
                                    revision_id,
                                    start_ts,
                                    text: text.clone(),
                                },
                            );
                            if live_transcript_debug {
                                tracing::debug!(
                                    user = %user_id,
                                    revision_id,
                                    transcript = %text,
                                    "streaming provisional transcription"
                                );
                            }
                        }

                        if recognizer.is_endpoint(&state.stream) {
                            let final_text = if !text.is_empty() {
                                text
                            } else {
                                state.last_partial_text.clone()
                            };
                            let final_text = polish_stream_final_text(&final_text);

                            if !final_text.is_empty() {
                                live_partial_text.remove(&(guild_id, user_id));
                                emissions.push(Utterance {
                                    user_id,
                                    start_ts,
                                    start_offset_ms,
                                    revision_id,
                                    stage: crate::app::UtteranceStage::StreamFinal,
                                    is_final: true,
                                    text: final_text.clone(),
                                    tokens,
                                    token_timestamps_s,
                                });
                                trim_trailing_silence_keep_tail(&mut state.pcm_16k, 3_200);
                                if offline_finalize_tx
                                    .try_send(OfflineFinalizeJob {
                                        user_id,
                                        start_ts,
                                        start_offset_ms,
                                        revision_id,
                                        stream_final_text: final_text.clone(),
                                        pcm_16k: std::mem::take(&mut state.pcm_16k),
                                    })
                                    .is_err()
                                {
                                    offline_finalize_dropped.fetch_add(1, AtomicOrdering::SeqCst);
                                } else {
                                    offline_finalize_inflight.fetch_add(1, AtomicOrdering::SeqCst);
                                }
                                if live_transcript_debug {
                                    tracing::debug!(
                                        user = %user_id,
                                        revision_id,
                                        transcript = %final_text,
                                        "streaming final transcription"
                                    );
                                }
                            }

                            recognizer.reset(&state.stream);
                            state.utterance_seq = state.utterance_seq.wrapping_add(1);
                            next_utterance_seq.insert(user_id, state.utterance_seq);
                            state.utterance_start_sample = None;
                            state.last_partial_text.clear();
                            state.last_emit_at = None;
                            state.dormant_after_endpoint = true;
                            state.dormant_silence_ticks = 0;
                            state.pcm_16k.clear();
                        }
                    }

                    for utterance in emissions {
                        pending_commits.fetch_add(1, AtomicOrdering::SeqCst);
                        if utterance_tx.blocking_send(utterance).is_err() {
                            pending_commits.fetch_sub(1, AtomicOrdering::SeqCst);
                        }
                    }
                }

                for user_id in to_drop {
                    live_partial_text.remove(&(guild_id, user_id));
                    streams.remove(&user_id);
                }
            }
            StreamingDecoderCommand::FlushAll {
                respond_to,
                observed_at,
            } => {
                let tracked_users = streams.keys().copied().collect::<Vec<_>>();
                for user_id in &tracked_users {
                    if let Some(state) = streams.get_mut(user_id) {
                        state.stream.input_finished();
                    }
                }

                let mut decode_iterations = 0usize;
                loop {
                    let ready_users = streams
                        .iter()
                        .filter_map(|(user_id, state)| recognizer.is_ready(&state.stream).then_some(*user_id))
                        .collect::<Vec<_>>();
                    if ready_users.is_empty() {
                        break;
                    }
                    decode_iterations += 1;
                    if decode_iterations > MAX_DECODE_BATCHES_PER_TICK {
                        tracing::warn!(guild = %guild_id, "streaming decode loop hit iteration cap during FlushAll");
                        break;
                    }

                    let ready_streams = ready_users
                        .iter()
                        .filter_map(|user_id| streams.get(user_id).map(|state| &state.stream))
                        .collect::<Vec<_>>();
                    recognizer.decode_multiple_streams(&ready_streams);
                }

                let mut emissions = Vec::<Utterance>::new();
                for user_id in tracked_users {
                    let Some(mut state) = streams.remove(&user_id) else {
                        continue;
                    };

                    let result = recognizer.get_result(&state.stream);
                    let result_text = result
                        .as_ref()
                        .map(|r| normalize_streaming_text(&r.text))
                        .unwrap_or_default();
                    let result_tokens = result
                        .as_ref()
                        .map(|r| r.tokens.clone())
                        .unwrap_or_default();
                    let result_token_timestamps_s = result
                        .as_ref()
                        .map(|r| r.timestamps.clone())
                        .flatten()
                        .unwrap_or_default();
                    let final_text = if !result_text.is_empty() {
                        result_text
                    } else {
                        state.last_partial_text.clone()
                    };
                    let final_text = polish_stream_final_text(&final_text);

                    live_partial_text.remove(&(guild_id, user_id));

                    if final_text.is_empty() {
                        continue;
                    }

                    let revision_id = make_revision_id(user_id, state.utterance_seq);
                    let start_sample = state
                        .utterance_start_sample
                        .unwrap_or(state.total_samples_fed);
                    let start_ts = state
                        .stream_anchor_at
                        .map(|anchor| sample_clock_to_instant(anchor, start_sample))
                        .unwrap_or(observed_at);
                    let start_offset_ms = sample_clock_to_offset_ms(start_sample);
                    emissions.push(Utterance {
                        user_id,
                        start_ts,
                        start_offset_ms,
                        revision_id,
                        stage: crate::app::UtteranceStage::StreamFinal,
                        is_final: true,
                        text: final_text.clone(),
                        tokens: result_tokens,
                        token_timestamps_s: result_token_timestamps_s,
                    });

                    trim_trailing_silence_keep_tail(&mut state.pcm_16k, 3_200);
                    if offline_finalize_tx
                        .try_send(OfflineFinalizeJob {
                            user_id,
                            start_ts,
                            start_offset_ms,
                            revision_id,
                            stream_final_text: final_text,
                            pcm_16k: std::mem::take(&mut state.pcm_16k),
                        })
                        .is_err()
                    {
                        offline_finalize_dropped.fetch_add(1, AtomicOrdering::SeqCst);
                    } else {
                        offline_finalize_inflight.fetch_add(1, AtomicOrdering::SeqCst);
                    }

                    next_utterance_seq.insert(user_id, state.utterance_seq.wrapping_add(1));
                }

                for utterance in emissions {
                    pending_commits.fetch_add(1, AtomicOrdering::SeqCst);
                    if utterance_tx.blocking_send(utterance).is_err() {
                        pending_commits.fetch_sub(1, AtomicOrdering::SeqCst);
                    }
                }

                let _ = respond_to.send(());
            }
        }
    }
        })
        .expect("failed to spawn streaming decoder thread")
}

pub async fn offline_finalize_worker_loop(
    offline_asr: Arc<AsrEngine>,
    mut rx: mpsc::Receiver<OfflineFinalizeJob>,
    utterance_tx: mpsc::Sender<Utterance>,
    pending_commits: Arc<AtomicUsize>,
    inflight: Arc<AtomicUsize>,
    refinement_rejected: Arc<AtomicUsize>,
    worker_alive: Arc<AtomicBool>,
    offline_rtf_milli_ewma: Arc<AtomicUsize>,
    offline_finalize_empty: Arc<AtomicUsize>,
    live_transcript_debug: bool,
) {
    struct OfflineWorkerAliveGuard {
        alive: Arc<AtomicBool>,
    }
    impl Drop for OfflineWorkerAliveGuard {
        fn drop(&mut self) {
            self.alive.store(false, AtomicOrdering::SeqCst);
        }
    }

    worker_alive.store(true, AtomicOrdering::SeqCst);
    let _alive_guard = OfflineWorkerAliveGuard {
        alive: Arc::clone(&worker_alive),
    };

    while let Some(job) = rx.recv().await {
        struct InflightTaskGuard {
            counter: Arc<AtomicUsize>,
        }
        impl Drop for InflightTaskGuard {
            fn drop(&mut self) {
                self.counter.fetch_sub(1, AtomicOrdering::SeqCst);
            }
        }

        let _guard = InflightTaskGuard {
            counter: Arc::clone(&inflight),
        };

        let audio_duration_ms = ((job.pcm_16k.len() as u64).saturating_mul(1000)) / 16_000;
        let Some(decoded) = transcribe_mono_pcm(Arc::clone(&offline_asr), job.pcm_16k).await else {
            continue;
        };

        if audio_duration_ms > 0 {
            let sample_rtf_milli = ((decoded.decode_elapsed_ms as f32 / audio_duration_ms as f32) * 1000.0)
                .round()
                .max(0.0) as usize;
            let mut prev = offline_rtf_milli_ewma.load(AtomicOrdering::Relaxed);
            loop {
                let next = if prev == 0 {
                    sample_rtf_milli
                } else {
                    ((prev * 8) + (sample_rtf_milli * 2) + 5) / 10
                };
                match offline_rtf_milli_ewma.compare_exchange_weak(
                    prev,
                    next,
                    AtomicOrdering::Relaxed,
                    AtomicOrdering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => prev = actual,
                }
            }
        }

        if decoded.text.trim().is_empty() {
            offline_finalize_empty.fetch_add(1, AtomicOrdering::SeqCst);
            continue;
        }

        if !should_accept_offline_final(&job.stream_final_text, &decoded.text) {
            refinement_rejected.fetch_add(1, AtomicOrdering::SeqCst);
            continue;
        }

        if live_transcript_debug {
            tracing::debug!(
                user = %job.user_id,
                revision_id = job.revision_id,
                transcript = %decoded.text,
                stream_final = %job.stream_final_text,
                "offline final transcription"
            );
        }

        pending_commits.fetch_add(1, AtomicOrdering::SeqCst);
        if utterance_tx
            .send(Utterance {
                user_id: job.user_id,
                start_ts: job.start_ts,
                start_offset_ms: job.start_offset_ms,
                revision_id: job.revision_id,
                stage: crate::app::UtteranceStage::OfflineFinal,
                is_final: true,
                text: decoded.text,
                tokens: decoded.tokens,
                token_timestamps_s: decoded.token_timestamps_s,
            })
            .await
            .is_err()
        {
            pending_commits.fetch_sub(1, AtomicOrdering::SeqCst);
        }
    }
}

fn texts_equivalent(a: &str, b: &str) -> bool {
    fn normalize_for_equivalence(input: &str) -> String {
        input
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .trim_end_matches(|c: char| matches!(c, '.' | '!' | '?' | ',' | ';' | ':'))
            .to_string()
    }

    normalize_for_equivalence(a).eq_ignore_ascii_case(&normalize_for_equivalence(b))
}

fn has_repeated_ngram(text: &str, n: usize, min_consecutive_repeats: usize) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < n.saturating_mul(min_consecutive_repeats) || n == 0 {
        return false;
    }

    let mut i = 0usize;
    while i + n <= words.len() {
        let candidate = &words[i..i + n];
        let mut repeats = 1usize;
        let mut j = i + n;
        while j + n <= words.len() && words[j..j + n].eq(candidate) {
            repeats += 1;
            if repeats >= min_consecutive_repeats {
                return true;
            }
            j += n;
        }
        i += 1;
    }

    false
}

pub type Streams = DashMap<(GuildId, UserId), UserStreamState>;

pub async fn transcript_writer_loop(
    session: Arc<RwLock<CallSession>>,
    mut rx: mpsc::Receiver<Utterance>,
    pending_commits: Arc<AtomicUsize>,
    transcript_jsonl_path: PathBuf,
) {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(Clone)]
    struct HeapItem {
        start_ts: Instant,
        seq: u64,
        utterance: Utterance,
    }

    impl Eq for HeapItem {}
    impl PartialEq for HeapItem {
        fn eq(&self, other: &Self) -> bool {
            self.start_ts == other.start_ts && self.seq == other.seq
        }
    }
    impl Ord for HeapItem {
        fn cmp(&self, other: &Self) -> Ordering {
            match self.start_ts.cmp(&other.start_ts) {
                Ordering::Equal => self.seq.cmp(&other.seq),
                o => o,
            }
            .reverse()
        }
    }
    impl PartialOrd for HeapItem {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut heap = BinaryHeap::new();
    let mut seq = 0u64;
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    let mut revision_index = std::collections::HashMap::<u64, usize>::new();

    #[derive(Serialize)]
    struct PersistedUtterance {
        revision_id: u64,
        user_id: u64,
        start_offset_ms: u64,
        stage: UtteranceStage,
        is_final: bool,
        text: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tokens: Vec<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        token_timestamps_s: Vec<f32>,
    }

    let mut journal_writer = fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&transcript_jsonl_path)
        .ok()
        .map(BufWriter::new);

    async fn apply_revision(
        session: &Arc<RwLock<CallSession>>,
        revision_index: &mut std::collections::HashMap<u64, usize>,
        pending_commits: &Arc<AtomicUsize>,
        journal_writer: &mut Option<BufWriter<fs::File>>,
        mut utterance: Utterance,
    ) {
        let mut lock = session.write().await;

        // Rebase to call-relative time at commit; decoder-side sample clocks are stream-local.
        let start_offset_ms = utterance
            .start_ts
            .saturating_duration_since(lock.started_mono)
            .as_millis() as u64;
        utterance.start_offset_ms = start_offset_ms;
        if !utterance.token_timestamps_s.is_empty() {
            let base_s = start_offset_ms as f32 / 1000.0;
            for t in &mut utterance.token_timestamps_s {
                *t += base_s;
            }
        }

        if revision_index.is_empty() || revision_index.len() != lock.transcript.len() {
            revision_index.clear();
            for (idx, u) in lock.transcript.iter().enumerate() {
                revision_index.insert(u.revision_id, idx);
            }
        }

        if let Some(existing_idx) = revision_index.get(&utterance.revision_id).copied() {
            if lock.transcript[existing_idx].stage.precedence() > utterance.stage.precedence() {
                pending_commits.fetch_sub(1, AtomicOrdering::SeqCst);
                return;
            }
            lock.transcript[existing_idx] = utterance;
            let persisted = (lock.transcript[existing_idx].stage != UtteranceStage::Partial)
                .then(|| PersistedUtterance {
                    revision_id: lock.transcript[existing_idx].revision_id,
                    user_id: lock.transcript[existing_idx].user_id.get(),
                    start_offset_ms: lock.transcript[existing_idx].start_offset_ms,
                    stage: lock.transcript[existing_idx].stage,
                    is_final: lock.transcript[existing_idx].is_final,
                    text: lock.transcript[existing_idx].text.clone(),
                    tokens: lock.transcript[existing_idx].tokens.clone(),
                    token_timestamps_s: lock.transcript[existing_idx].token_timestamps_s.clone(),
                });
            drop(lock);
            if let (Some(writer), Some(item)) = (journal_writer.as_mut(), persisted) {
                if let Ok(line) = serde_json::to_string(&item) {
                    let _ = writeln!(writer, "{line}");
                    let _ = writer.flush();
                }
            }
            pending_commits.fetch_sub(1, AtomicOrdering::SeqCst);
            return;
        }

        lock.transcript.push(utterance.clone());
        lock.transcript.sort_by_key(|u| u.start_ts);
        revision_index.clear();
        for (idx, u) in lock.transcript.iter().enumerate() {
            revision_index.insert(u.revision_id, idx);
        }

        let persisted = (utterance.stage != UtteranceStage::Partial).then(|| PersistedUtterance {
            revision_id: utterance.revision_id,
            user_id: utterance.user_id.get(),
            start_offset_ms: utterance.start_offset_ms,
            stage: utterance.stage,
            is_final: utterance.is_final,
            text: utterance.text,
            tokens: utterance.tokens,
            token_timestamps_s: utterance.token_timestamps_s,
        });

        drop(lock);
        if let (Some(writer), Some(item)) = (journal_writer.as_mut(), persisted) {
            if let Ok(line) = serde_json::to_string(&item) {
                let _ = writeln!(writer, "{line}");
                let _ = writer.flush();
            }
        }

        pending_commits.fetch_sub(1, AtomicOrdering::SeqCst);
    }

    loop {
        tokio::select! {
            maybe_u = rx.recv() => {
                match maybe_u {
                    Some(u) => {
                        heap.push(HeapItem { start_ts: u.start_ts, seq, utterance: u });
                        seq = seq.wrapping_add(1);
                    }
                    None => break,
                }
            }
            _ = tick.tick() => {}
        }

        let watermark = Instant::now() - REORDER_WINDOW;
        while let Some(top) = heap.peek() {
            if top.start_ts > watermark {
                break;
            }

            if let Some(item) = heap.pop() {
                apply_revision(
                    &session,
                    &mut revision_index,
                    &pending_commits,
                    &mut journal_writer,
                    item.utterance,
                )
                    .await;
            }
        }
    }

    while let Some(item) = heap.pop() {
        apply_revision(
            &session,
            &mut revision_index,
            &pending_commits,
            &mut journal_writer,
            item.utterance,
        )
        .await;
    }
}

pub type SsrcMap = DashMap<(GuildId, u32), UserId>;
pub type SessionSenders = DashMap<GuildId, mpsc::Sender<Utterance>>;
pub type StreamingDecoderSenders = DashMap<GuildId, mpsc::Sender<StreamingDecoderCommand>>;
