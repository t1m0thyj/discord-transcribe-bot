use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use dashmap::DashMap;
use earshot::Detector;
use nnnoiseless::DenoiseState;
use serenity::all::{GuildId, UserId};
use sherpa_onnx::{
    OfflineMoonshineModelConfig, OfflineNemoEncDecCtcModelConfig, OfflineParaformerModelConfig,
    OfflineQwen3ASRModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSenseVoiceModelConfig,
    OfflineTdnnModelConfig, OfflineTransducerModelConfig, OfflineWhisperModelConfig,
    OfflineZipformerCtcModelConfig,
};
use tokio::sync::{mpsc, RwLock};
use tokio::sync::Semaphore;

use crate::app::{CallSession, Utterance};

pub const SILENCE_TICKS_THRESHOLD: u32 = 20;
pub const REORDER_WINDOW: Duration = Duration::from_millis(1500);
pub const RNNOISE_FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;
const EARSHOT_FRAME_SIZE: usize = 256;
const EARSHOT_VAD_THRESHOLD: f32 = 0.5;
const VAD_HANGOVER_FRAMES: u8 = 8;
const AGC_TARGET_RMS: f32 = 0.06;
const AGC_MIN_GAIN: f32 = 0.6;
const AGC_MAX_GAIN: f32 = 3.0;
const DENOISER_BYPASS_SNR_DB: f32 = 18.0;
const HIGHPASS_ALPHA_48K: f32 = 0.991;

pub struct AsrEngine {
    recognizer: Arc<OfflineRecognizer>,
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
    pub fn new(model_dir: &str) -> anyhow::Result<Self> {
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

        cfg.model_config.num_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .clamp(1, 8);

        let recognizer = OfflineRecognizer::create(&cfg).ok_or_else(|| {
            anyhow::anyhow!(
                "sherpa-onnx failed to create a recognizer for the {} backend from {}",
                selected_backend,
                model_base.display()
            )
        })?;

        tracing::info!(
            "ASR backend selected: {} (ASR_MODEL_DIR={})",
            selected_backend,
            model_base.display()
        );

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
    vad: Detector,
    pending: Vec<f32>,
    vad_pending_16k: Vec<f32>,
    warmed_up: bool,
    agc_gain: f32,
    hp_prev_x: f32,
    hp_prev_y: f32,
    noise_rms_ema: f32,
    last_snr_db: f32,
    resample_accum: f32,
    resample_phase: u8,
    vad_hangover_frames: u8,
    was_speech_last_tick: bool,
}

pub struct ProcessedSpeechChunk {
    pub pcm_16k: Vec<f32>,
    pub speech_active: bool,
}

impl UserDenoiseState {
    pub fn new() -> Self {
        Self {
            denoiser: DenoiseState::new(),
            vad: Detector::default(),
            pending: Vec::new(),
            vad_pending_16k: Vec::new(),
            warmed_up: false,
            agc_gain: 1.0,
            hp_prev_x: 0.0,
            hp_prev_y: 0.0,
            noise_rms_ema: 0.002,
            last_snr_db: 0.0,
            resample_accum: 0.0,
            resample_phase: 0,
            vad_hangover_frames: 0,
            was_speech_last_tick: false,
        }
    }

    pub fn push_stereo_pcm(&mut self, input: &[i16], enable_denoiser: bool) -> ProcessedSpeechChunk {
        let mut mono = downmix_stereo_to_mono_unit_scale(input);
        if mono.is_empty() {
            return ProcessedSpeechChunk {
                pcm_16k: Vec::new(),
                speech_active: false,
            };
        }

        self.apply_highpass(&mut mono);

        let rms = compute_rms(&mono);
        self.update_noise_and_snr(rms);
        let use_denoiser = enable_denoiser && self.last_snr_db < DENOISER_BYPASS_SNR_DB;
        let mut cleaned_48k = if use_denoiser {
            self.push_mono_pcm(&mono)
        } else {
            self.pending.clear();
            mono
        };

        let cleaned_rms = compute_rms(&cleaned_48k);
        self.apply_agc(&mut cleaned_48k, cleaned_rms);

        let pcm_16k = self.resample_48k_to_16k_stream(&cleaned_48k);
        let speech_active = self.passes_vad_earshot(&pcm_16k);
        self.was_speech_last_tick = speech_active;

        ProcessedSpeechChunk {
            pcm_16k,
            speech_active,
        }
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

    pub fn flush_pending(&mut self) -> Vec<f32> {
        if self.pending.is_empty() {
            return Vec::new();
        }

        let mut frame = std::mem::take(&mut self.pending);
        frame.resize(RNNOISE_FRAME_SIZE, 0.0);
        let cleaned = self.denoise_frame(&frame);
        self.resample_48k_to_16k_stream(&cleaned)
    }

    fn update_noise_and_snr(&mut self, rms: f32) {
        let noise_update = if rms <= self.noise_rms_ema * 1.5 { 0.08 } else { 0.005 };
        self.noise_rms_ema = self.noise_rms_ema * (1.0 - noise_update) + rms * noise_update;
        let noise = self.noise_rms_ema.max(1e-4);
        self.last_snr_db = 20.0 * ((rms + 1e-4) / noise).log10();
    }

    fn passes_vad_earshot(&mut self, pcm_16k: &[f32]) -> bool {
        if !pcm_16k.is_empty() {
            self.vad_pending_16k.extend_from_slice(pcm_16k);
        }

        let mut evaluated_any = false;
        let mut voiced = false;
        while self.vad_pending_16k.len() >= EARSHOT_FRAME_SIZE {
            let frame: Vec<f32> = self.vad_pending_16k.drain(..EARSHOT_FRAME_SIZE).collect();
            let score = self.vad.predict_f32(&frame);
            evaluated_any = true;
            if score >= EARSHOT_VAD_THRESHOLD {
                voiced = true;
            }
        }

        if voiced {
            self.vad_hangover_frames = VAD_HANGOVER_FRAMES;
            return true;
        }

        if evaluated_any {
            if self.vad_hangover_frames > 0 {
                self.vad_hangover_frames -= 1;
                return true;
            }
            return false;
        }

        self.vad_hangover_frames > 0
    }

    fn apply_agc(&mut self, samples: &mut [f32], rms: f32) {
        if !self.was_speech_last_tick {
            for sample in samples {
                *sample = (*sample * self.agc_gain).clamp(-0.98, 0.98);
            }
            return;
        }

        let desired_gain = (AGC_TARGET_RMS / rms.max(1e-4)).clamp(AGC_MIN_GAIN, AGC_MAX_GAIN);
        let smoothing = if desired_gain < self.agc_gain { 0.35 } else { 0.02 };
        self.agc_gain = self.agc_gain * (1.0 - smoothing) + desired_gain * smoothing;

        for sample in samples {
            *sample = (*sample * self.agc_gain).clamp(-0.98, 0.98);
        }
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

        let mut out = Vec::with_capacity(input.len() / 3 + 1);
        for &sample in input {
            self.resample_accum += sample;
            self.resample_phase = self.resample_phase.wrapping_add(1);
            if self.resample_phase == 3 {
                out.push(self.resample_accum / 3.0);
                self.resample_phase = 0;
                self.resample_accum = 0.0;
            }
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

#[derive(Default, Clone)]
pub struct UserAudioBuffer {
    pub pcm: Vec<f32>,
    pub silent_ticks: u32,
    pub utterance_start: Option<Instant>,
    pub current_revision_seq: Option<u64>,
    pub next_revision_seq: u64,
    pub last_preview_samples: usize,
    pub last_preview_text: Option<String>,
    pub frozen_prefix_words: usize,
    pub stable_preview_streak: u32,
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

pub async fn transcribe_mono_pcm(
    asr: Arc<AsrEngine>,
    pcm_mono: Vec<f32>,
) -> Option<String> {
    if pcm_mono.len() < 1600 {
        return None;
    }

    // Bound concurrent ASR decode work to keep CPU usage stable on small devices.
    let permit = asr_decode_semaphore().acquire_owned().await.ok()?;

    let text = tokio::task::spawn_blocking(move || asr.transcribe_16k_mono(&pcm_mono))
        .await
        .ok()?;
    drop(permit);

    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

fn asr_decode_semaphore() -> Arc<Semaphore> {
    static ASR_DECODE_SEMAPHORE: OnceLock<Arc<Semaphore>> = OnceLock::new();
    Arc::clone(ASR_DECODE_SEMAPHORE.get_or_init(|| Arc::new(Semaphore::new(1))))
}

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

pub type Denoisers = DashMap<(GuildId, UserId), UserDenoiseState>;

pub async fn transcript_writer_loop(
    session: Arc<RwLock<CallSession>>,
    mut rx: mpsc::Receiver<Utterance>,
    pending_commits: Arc<AtomicUsize>,
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

    async fn apply_revision(
        session: &Arc<RwLock<CallSession>>,
        revision_index: &mut std::collections::HashMap<u64, usize>,
        pending_commits: &Arc<AtomicUsize>,
        utterance: Utterance,
    ) {
        let mut lock = session.write().await;

        if revision_index.is_empty() || revision_index.len() != lock.transcript.len() {
            revision_index.clear();
            for (idx, u) in lock.transcript.iter().enumerate() {
                revision_index.insert(u.revision_id, idx);
            }
        }

        if let Some(existing_idx) = revision_index.get(&utterance.revision_id).copied() {
            if lock.transcript[existing_idx].is_final && !utterance.is_final {
                pending_commits.fetch_sub(1, AtomicOrdering::SeqCst);
                return;
            }
            lock.transcript[existing_idx] = utterance;
            pending_commits.fetch_sub(1, AtomicOrdering::SeqCst);
            return;
        }

        lock.transcript.push(utterance.clone());
        lock.transcript.sort_by_key(|u| u.start_ts);
        revision_index.clear();
        for (idx, u) in lock.transcript.iter().enumerate() {
            revision_index.insert(u.revision_id, idx);
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
                apply_revision(&session, &mut revision_index, &pending_commits, item.utterance)
                    .await;
            }
        }
    }

    while let Some(item) = heap.pop() {
        apply_revision(&session, &mut revision_index, &pending_commits, item.utterance).await;
    }
}

pub type Buffers = DashMap<(GuildId, UserId), UserAudioBuffer>;
pub type SsrcMap = DashMap<(GuildId, u32), UserId>;
pub type SessionSenders = DashMap<GuildId, mpsc::Sender<Utterance>>;
