use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context as _;
use dashmap::DashMap;
use serenity::all::{GuildId, UserId};
use sherpa_onnx::{
    OfflineMoonshineModelConfig, OfflineNemoEncDecCtcModelConfig, OfflineParaformerModelConfig,
    OfflineQwen3ASRModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSenseVoiceModelConfig,
    OfflineTdnnModelConfig, OfflineTransducerModelConfig, OfflineWhisperModelConfig,
    OfflineZipformerCtcModelConfig,
};
use tokio::sync::{mpsc, RwLock};
use tokio::io::AsyncWriteExt;
use serde::Serialize;

use crate::app::{CallSession, Utterance};
use crate::denoiser::{compute_rms, UserDenoiseState};

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
    // On explicit export/finalize, silence_ticks can be 0 while speech is still active.
    // In that case we skip trimming to avoid clipping live speech tails.
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

/// Explicit override for the single-file model families that are indistinguishable
/// from filenames alone (Paraformer / SenseVoice / NeMo CTC / Zipformer CTC / TDNN
/// are all typically just "model.onnx" + "tokens.txt").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForcedFamily {
    Paraformer,
    SenseVoice,
    NemoCtc,
    ZipformerCtc,
    Tdnn,
}

impl ForcedFamily {
    fn from_hint(raw: &str) -> Option<Self> {
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
    pub fn new(
        model_dir: &str,
        asr_num_threads: i32,
        forced_family_hint: Option<&str>,
    ) -> anyhow::Result<Self> {
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
        } else if let Some(label) = try_single_file_family(&mut cfg, &model_base, forced_family_hint)? {
            label
        } else {
            anyhow::bail!(
                "could not identify a supported ASR model family in {} \
                 (looked for transducer encoder/decoder/joiner, Whisper encoder/decoder, \
                 Moonshine's split or merged files, and single-file model.onnx variants)",
                model_base.display()
            );
        };

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
    forced_family_hint: Option<&str>,
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

    let family = forced_family_hint
        .and_then(ForcedFamily::from_hint)
        .or_else(|| ForcedFamily::guess_from_dir_name(dir))
        .ok_or_else(|| anyhow::anyhow!(
            "found a single model.onnx in {} but can't tell which family it is -- \
             Paraformer, SenseVoice, NeMo CTC, Zipformer CTC, and TDNN models are all \
             shipped this way and are not distinguishable by filename alone. \
             Set [asr].model_family in config.toml to one of: paraformer, sense_voice, nemo_ctc, zipformer_ctc, tdnn",
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

    tracing::debug!(
        samples = sample_count,
        decode_ms,
        "asr decode completed"
    );

    let text = text.trim().to_string();
    if text.is_empty() {
        return None;
    }

    let audio_secs = (sample_count as f32 / 16_000.0).max(0.001);
    if let Some(reason) = decode_rejection_reason(&text, audio_secs) {
        tracing::warn!(
            reason,
            samples = sample_count,
            "rejected decoded utterance"
        );
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

pub type Streams = DashMap<(GuildId, UserId), UserStreamState>;

pub async fn transcript_writer_loop(
    session: Arc<RwLock<CallSession>>,
    mut rx: mpsc::Receiver<Utterance>,
    pending_commits: Arc<AtomicUsize>,
    transcript_jsonl_path: PathBuf,
) {
    #[derive(Serialize)]
    struct PersistedUtterance {
        user_id: u64,
        start_offset_ms: u64,
        text: String,
    }

    async fn append_persisted_utterance(path: &Path, item: &PersistedUtterance) {
        let Ok(mut file) = tokio::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
            .await
        else {
            return;
        };
        let Ok(line) = serde_json::to_string(item) else {
            return;
        };
        let _ = file.write_all(format!("{line}\n").as_bytes()).await;
    }

    async fn append_utterance(
        session: &Arc<RwLock<CallSession>>,
        pending_commits: &Arc<AtomicUsize>,
        transcript_jsonl_path: &Path,
        utterance: Utterance,
    ) {
        let mut lock = session.write().await;

        let start_offset_ms = utterance
            .start_ts
            .saturating_duration_since(lock.started_mono)
            .as_millis() as u64;
        lock.transcript.push(utterance.clone());

        let persisted = PersistedUtterance {
            user_id: utterance.user_id.get(),
            start_offset_ms,
            text: utterance.text,
        };

        drop(lock);
        append_persisted_utterance(transcript_jsonl_path, &persisted).await;

        pending_commits.fetch_sub(1, AtomicOrdering::SeqCst);
    }

    while let Some(item) = rx.recv().await {
        append_utterance(
            &session,
            &pending_commits,
            &transcript_jsonl_path,
            item,
        )
        .await;
    }
}

pub type SsrcMap = DashMap<(GuildId, u32), UserId>;
