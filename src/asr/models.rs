use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use sherpa_onnx::{
    OfflineMoonshineModelConfig, OfflineNemoEncDecCtcModelConfig, OfflineParaformerModelConfig,
    OfflineQwen3ASRModelConfig, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig,
    OfflineTdnnModelConfig, OfflineTransducerModelConfig, OfflineWhisperModelConfig,
    OfflineZipformerCtcModelConfig,
};

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

pub(super) fn resolve_model_dir(model_dir: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(model_dir);
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }

    let cwd = std::env::current_dir().context("failed to read current working directory")?;
    Ok(cwd.join(path))
}

pub(super) fn configure_model(
    cfg: &mut OfflineRecognizerConfig,
    model_base: &Path,
    forced_family_hint: Option<&str>,
) -> anyhow::Result<&'static str> {
    // Order is significant: try the most specific layouts before broader heuristics.
    // In particular, qwen3_asr (conv_frontend + tokenizer/) must run before whisper,
    // because whisper matching only looks for encoder/decoder style names.
    if let Some(label) = try_transducer(cfg, model_base) {
        return Ok(label);
    }
    if let Some(label) = try_moonshine(cfg, model_base) {
        return Ok(label);
    }
    if let Some(label) = try_qwen3_asr(cfg, model_base) {
        return Ok(label);
    }
    if let Some(label) = try_whisper(cfg, model_base)? {
        return Ok(label);
    }
    if let Some(label) = try_single_file_family(cfg, model_base, forced_family_hint)? {
        return Ok(label);
    }

    anyhow::bail!(
        "could not identify a supported ASR model family in {} \
         (looked for transducer encoder/decoder/joiner, Whisper encoder/decoder, \
         Moonshine's split or merged files, and single-file model.onnx variants)",
        model_base.display()
    )
}

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
        (!is_int8, p.clone())
    });

    candidates.into_iter().next()
}

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
    let merged = find_by_prefix(dir, "merged_decod");
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
    let Some(encoder) = find_by_hint(dir, "encoder").or_else(|| find_by_hint(dir, "large")) else {
        return Ok(None);
    };
    let Some(decoder) = find_by_hint(dir, "decoder").or_else(|| find_by_hint(dir, "large")) else {
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use sherpa_onnx::OfflineRecognizerConfig;

    use super::{ForcedFamily, configure_model};

    fn test_temp_dir(name: &str) -> PathBuf {
        let unique = format!(
            "transcribe-bot-tests-{}-{}-{}",
            name,
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("valid clock")
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn forced_family_parses_common_aliases() {
        assert_eq!(ForcedFamily::from_hint("sense-voice"), Some(ForcedFamily::SenseVoice));
        assert_eq!(ForcedFamily::from_hint("zipformer_ctc"), Some(ForcedFamily::ZipformerCtc));
        assert_eq!(ForcedFamily::from_hint("nemoctc"), Some(ForcedFamily::NemoCtc));
    }

    #[test]
    fn forced_family_guesses_from_directory_name() {
        assert_eq!(
            ForcedFamily::guess_from_dir_name(PathBuf::from("model-sensevoice").as_path()),
            Some(ForcedFamily::SenseVoice)
        );
        assert_eq!(
            ForcedFamily::guess_from_dir_name(PathBuf::from("acoustic-tdnn").as_path()),
            Some(ForcedFamily::Tdnn)
        );
    }

    #[test]
    fn configure_model_supports_single_file_with_forced_family() {
        let dir = test_temp_dir("single-file-forced-family");
        fs::write(dir.join("model.onnx"), b"").expect("write model");
        fs::write(dir.join("tokens.txt"), b"").expect("write tokens");

        let mut cfg = OfflineRecognizerConfig::default();
        let label = configure_model(&mut cfg, &dir, Some("paraformer"))
            .expect("model should configure");

        assert_eq!(label, "paraformer");
        assert!(cfg.model_config.paraformer.model.is_some());
        assert!(cfg.model_config.tokens.is_some());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn configure_model_single_file_without_family_errors() {
        let dir = test_temp_dir("single-file-missing-family");
        fs::write(dir.join("model.int8.onnx"), b"").expect("write model");
        fs::write(dir.join("tokens.txt"), b"").expect("write tokens");

        let mut cfg = OfflineRecognizerConfig::default();
        let err = configure_model(&mut cfg, &dir, None)
            .expect_err("single-file config should fail without family hint");
        assert!(err.to_string().contains("can't tell which family"));

        let _ = fs::remove_dir_all(&dir);
    }
}
