use std::env;
use std::path::PathBuf;

use serde::Deserialize;

use crate::ai::AiProviderConfig;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub discord_token: String,
    pub ai: AiRuntimeConfig,
    pub asr: AsrRuntimeConfig,
    pub audio: AudioRuntimeConfig,
    pub debug: DebugRuntimeConfig,
    pub transcription: TranscriptionRuntimeConfig,
    pub discord: DiscordRuntimeConfig,
    pub summary: SummaryRuntimeConfig,
}

#[derive(Clone, Debug)]
pub struct AiRuntimeConfig {
    pub provider: AiProviderConfig,
    pub request_timeout: u64,
}

#[derive(Clone, Debug)]
pub struct AsrRuntimeConfig {
    pub model_dir: String,
    pub model_family: Option<String>,
    pub num_threads: i32,
}

#[derive(Clone, Debug)]
pub struct AudioRuntimeConfig {
    pub enable_denoiser: bool,
}

#[derive(Clone, Debug)]
pub struct DebugRuntimeConfig {
    pub log_live_transcript: bool,
}

#[derive(Clone, Debug)]
pub struct TranscriptionRuntimeConfig {
    pub endpoint_silence_ms: u64,
    pub rolling_ingest_max_ms: u64,
    pub rolling_ingest_context_ms: u64,
}

#[derive(Clone, Debug)]
pub struct DiscordRuntimeConfig {
    pub autojoin_suffix: String,
}

#[derive(Clone, Debug)]
pub struct SummaryRuntimeConfig {
    pub enabled: bool,
    pub post_in_thread: bool,
    pub include_in_markdown: bool,
    pub timeout_secs: u64,
}

impl AppConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let file_cfg = load_file_config()?;

        let discord_token = env::var("DISCORD_TOKEN")?;
        let ai_provider_name = file_cfg
            .ai
            .as_ref()
            .and_then(|ai| ai.provider.clone())
            .unwrap_or_else(|| "ollama".to_string());
        let gemini_model = file_cfg
            .ai
            .as_ref()
            .and_then(|ai| ai.gemini.as_ref().and_then(|g| g.model.clone()));
        let ollama_model = file_cfg
            .ai
            .as_ref()
            .and_then(|ai| ai.ollama.as_ref().and_then(|o| o.model.clone()));
        let ollama_base_url = file_cfg
            .ai
            .as_ref()
            .and_then(|ai| ai.ollama.as_ref().and_then(|o| o.base_url.clone()));
        let ai_provider = AiProviderConfig::from_selection(
            &ai_provider_name,
            read_nonempty_env("GEMINI_API_KEY"),
            gemini_model,
            ollama_model,
            ollama_base_url,
        )?;
        let ai_request_timeout = file_cfg
            .ai
            .as_ref()
            .and_then(|ai| ai.request_timeout)
            .filter(|v| *v >= 5)
            .unwrap_or(30);

        let asr_model_dir = file_cfg
            .asr
            .as_ref()
            .and_then(|asr| asr.model_dir.clone())
            .ok_or_else(|| anyhow::anyhow!("missing ASR model directory: set [asr].model_dir in config.toml"))?;
        let asr_model_family = file_cfg
            .asr
            .as_ref()
            .and_then(|asr| asr.model_family.clone());
        let detected_cores = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .clamp(1, 8);
        let default_asr_num_threads = if detected_cores >= 4 { 3 } else { detected_cores };
        let asr_num_threads = file_cfg
            .asr
            .as_ref()
            .and_then(|asr| asr.num_threads)
            .filter(|v| *v >= 1)
            .unwrap_or(default_asr_num_threads)
            .clamp(1, 8);
        let log_live_transcript = file_cfg
            .debug
            .as_ref()
            .and_then(|d| d.log_live_transcript)
            .unwrap_or(false);
        let enable_denoiser = file_cfg
            .audio
            .as_ref()
            .and_then(|a| a.enable_denoiser)
            .unwrap_or(false);
        let endpoint_silence_ms = file_cfg
            .transcription
            .as_ref()
            .and_then(|t| t.endpoint_silence_ms)
            .filter(|v| *v >= 80)
            .unwrap_or(400);

        // Keep in-memory audio bounded for long uninterrupted speech.
        let rolling_ingest_max_ms = file_cfg
            .transcription
            .as_ref()
            .and_then(|t| t.rolling_ingest_max_ms)
            .filter(|v| *v >= 4_000)
            .unwrap_or(12_000);

        let rolling_ingest_context_ms = file_cfg
            .transcription
            .as_ref()
            .and_then(|t| t.rolling_ingest_context_ms)
            .filter(|v| *v >= 250)
            .unwrap_or(1_500);

        let autojoin_suffix = file_cfg
            .discord
            .as_ref()
            .and_then(|d| d.autojoin_suffix.clone())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "[Transcribe]".to_string());

        let post_call_summary_enabled = file_cfg
            .summary
            .as_ref()
            .and_then(|s| s.enabled)
            .unwrap_or(false);

        let post_call_summary_post_in_thread = file_cfg
            .summary
            .as_ref()
            .and_then(|s| s.post_in_thread)
            .unwrap_or(true);

        let post_call_summary_include_in_markdown = file_cfg
            .summary
            .as_ref()
            .and_then(|s| s.include_in_markdown)
            .unwrap_or(false);

        let post_call_summary_timeout_secs = file_cfg
            .summary
            .as_ref()
            .and_then(|s| s.timeout_secs)
            .filter(|v| *v >= 5)
            .unwrap_or(25);

        Ok(Self {
            discord_token,
            ai: AiRuntimeConfig {
                provider: ai_provider,
                request_timeout: ai_request_timeout,
            },
            asr: AsrRuntimeConfig {
                model_dir: asr_model_dir,
                model_family: asr_model_family,
                num_threads: asr_num_threads,
            },
            audio: AudioRuntimeConfig { enable_denoiser },
            debug: DebugRuntimeConfig {
                log_live_transcript,
            },
            transcription: TranscriptionRuntimeConfig {
                endpoint_silence_ms,
                rolling_ingest_max_ms,
                rolling_ingest_context_ms,
            },
            discord: DiscordRuntimeConfig { autojoin_suffix },
            summary: SummaryRuntimeConfig {
                enabled: post_call_summary_enabled,
                post_in_thread: post_call_summary_post_in_thread,
                include_in_markdown: post_call_summary_include_in_markdown,
                timeout_secs: post_call_summary_timeout_secs,
            },
        })
    }
}

#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    ai: Option<AiSection>,
    asr: Option<AsrSection>,
    audio: Option<AudioSection>,
    debug: Option<DebugSection>,
    transcription: Option<TranscriptionSection>,
    discord: Option<DiscordSection>,
    summary: Option<SummarySection>,
}

#[derive(Debug, Default, Deserialize)]
struct AiSection {
    provider: Option<String>,
    request_timeout: Option<u64>,
    ollama: Option<OllamaSection>,
    gemini: Option<GeminiSection>,
}

#[derive(Debug, Default, Deserialize)]
struct OllamaSection {
    model: Option<String>,
    base_url: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct GeminiSection {
    model: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct AsrSection {
    model_dir: Option<String>,
    model_family: Option<String>,
    num_threads: Option<i32>,
}

#[derive(Debug, Default, Deserialize)]
struct AudioSection {
    enable_denoiser: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct TranscriptionSection {
    endpoint_silence_ms: Option<u64>,
    rolling_ingest_max_ms: Option<u64>,
    rolling_ingest_context_ms: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
struct DebugSection {
    log_live_transcript: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct DiscordSection {
    autojoin_suffix: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct SummarySection {
    enabled: Option<bool>,
    post_in_thread: Option<bool>,
    include_in_markdown: Option<bool>,
    timeout_secs: Option<u64>,
}

fn load_file_config() -> anyhow::Result<FileConfig> {
    let config_path = resolve_config_path();
    let path = PathBuf::from(config_path);
    if !path.is_file() {
        return Ok(FileConfig::default());
    }

    let raw = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {}", path.display(), e))?;
    let cfg = toml::from_str::<FileConfig>(&raw)
        .map_err(|e| anyhow::anyhow!("invalid TOML in {}: {}", path.display(), e))?;
    Ok(cfg)
}

fn resolve_config_path() -> String {
    if let Some(path) = read_nonempty_env("APP_CONFIG_PATH") {
        return path;
    }
    "config.toml".to_string()
}

fn read_nonempty_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

