use std::env;
use std::path::PathBuf;

use anyhow::Context as _;
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
    pub retention_days: u64,
}

#[derive(Clone, Debug)]
pub struct DiscordRuntimeConfig {
    pub autojoin_suffix: String,
    pub autojoin_text_channel_id: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct SummaryRuntimeConfig {
    pub enabled: bool,
    pub post_in_thread: bool,
    pub include_in_markdown: bool,
}

impl AppConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let file_cfg = load_file_config()?;

        let discord_token = required_nonempty_env("DISCORD_TOKEN")?;
        let openai_model = file_cfg.ai.as_ref().and_then(|ai| ai.model.clone());
        let openai_base_url = file_cfg.ai.as_ref().and_then(|ai| ai.base_url.clone());
        let api_key = resolve_ai_api_key(file_cfg.ai.as_ref())?;
        let ai_provider =
            AiProviderConfig::openai_compatible(api_key, openai_model, openai_base_url)?;
        let ai_request_timeout = file_cfg
            .ai
            .as_ref()
            .and_then(|ai| ai.request_timeout)
            .filter(|v| *v >= 5)
            .unwrap_or(30);

        let detected_cores = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .clamp(1, 8);
        let asr = resolve_asr_runtime_config(file_cfg.asr.as_ref(), detected_cores)?;
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
        let transcription = resolve_transcription_runtime_config(file_cfg.transcription.as_ref());
        let autojoin_suffix = resolve_autojoin_suffix(file_cfg.discord.as_ref());
        let autojoin_text_channel_id = resolve_autojoin_text_channel_id(file_cfg.discord.as_ref())?;

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

        Ok(Self {
            discord_token,
            ai: AiRuntimeConfig {
                provider: ai_provider,
                request_timeout: ai_request_timeout,
            },
            asr,
            audio: AudioRuntimeConfig { enable_denoiser },
            debug: DebugRuntimeConfig {
                log_live_transcript,
            },
            transcription,
            discord: DiscordRuntimeConfig {
                autojoin_suffix,
                autojoin_text_channel_id,
            },
            summary: SummaryRuntimeConfig {
                enabled: post_call_summary_enabled,
                post_in_thread: post_call_summary_post_in_thread,
                include_in_markdown: post_call_summary_include_in_markdown,
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
    request_timeout: Option<u64>,
    model: Option<String>,
    base_url: Option<String>,
    api_key_env: Option<String>,
}

fn resolve_ai_api_key(section: Option<&AiSection>) -> anyhow::Result<Option<String>> {
    let api_key_env = section
        .and_then(|ai| ai.api_key_env.as_deref())
        .map(str::trim)
        .filter(|name| !name.is_empty());
    api_key_env
        .map(required_nonempty_env)
        .transpose()
        .with_context(|| "reading [ai].api_key_env")
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
    retention_days: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
struct DebugSection {
    log_live_transcript: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct DiscordSection {
    autojoin_suffix: Option<String>,
    autojoin_text_channel_id: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
struct SummarySection {
    enabled: Option<bool>,
    post_in_thread: Option<bool>,
    include_in_markdown: Option<bool>,
}

fn resolve_asr_runtime_config(
    section: Option<&AsrSection>,
    detected_cores: i32,
) -> anyhow::Result<AsrRuntimeConfig> {
    let model_dir = section
        .and_then(|asr| asr.model_dir.clone())
        .ok_or_else(|| {
            anyhow::anyhow!("missing ASR model directory: set [asr].model_dir in config.toml")
        })?;
    let default_threads = if detected_cores >= 4 {
        3
    } else {
        detected_cores
    };
    let num_threads = section
        .and_then(|asr| asr.num_threads)
        .filter(|threads| *threads >= 1)
        .unwrap_or(default_threads)
        .clamp(1, 8);

    Ok(AsrRuntimeConfig {
        model_dir,
        model_family: section.and_then(|asr| asr.model_family.clone()),
        num_threads,
    })
}

fn resolve_transcription_runtime_config(
    section: Option<&TranscriptionSection>,
) -> TranscriptionRuntimeConfig {
    TranscriptionRuntimeConfig {
        endpoint_silence_ms: section
            .and_then(|transcription| transcription.endpoint_silence_ms)
            .filter(|milliseconds| *milliseconds >= 80)
            .unwrap_or(400),
        rolling_ingest_max_ms: section
            .and_then(|transcription| transcription.rolling_ingest_max_ms)
            .filter(|milliseconds| *milliseconds >= 4_000)
            .unwrap_or(12_000),
        rolling_ingest_context_ms: section
            .and_then(|transcription| transcription.rolling_ingest_context_ms)
            .filter(|milliseconds| *milliseconds >= 250)
            .unwrap_or(1_500),
        retention_days: section
            .and_then(|transcription| transcription.retention_days)
            .filter(|days| *days >= 1)
            .unwrap_or(30),
    }
}

fn resolve_autojoin_suffix(section: Option<&DiscordSection>) -> String {
    section
        .and_then(|discord| discord.autojoin_suffix.clone())
        .filter(|suffix| !suffix.trim().is_empty())
        .unwrap_or_else(|| "[Transcribe]".to_string())
}

fn resolve_autojoin_text_channel_id(
    section: Option<&DiscordSection>,
) -> anyhow::Result<Option<u64>> {
    match section.and_then(|discord| discord.autojoin_text_channel_id) {
        Some(0) => anyhow::bail!(
            "[discord].autojoin_text_channel_id must be a non-zero Discord channel ID"
        ),
        channel_id => Ok(channel_id),
    }
}

fn load_file_config() -> anyhow::Result<FileConfig> {
    let config_path = resolve_config_path();
    if !config_path.is_file() {
        return Ok(FileConfig::default());
    }

    let raw = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {}", config_path.display(), e))?;
    let cfg = toml::from_str::<FileConfig>(&raw)
        .map_err(|e| anyhow::anyhow!("invalid TOML in {}: {}", config_path.display(), e))?;
    Ok(cfg)
}

pub(crate) fn resolve_config_path() -> PathBuf {
    if let Some(path) = read_nonempty_env("APP_CONFIG_PATH") {
        return PathBuf::from(path);
    }
    PathBuf::from("config.toml")
}

fn read_nonempty_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn required_nonempty_env(name: &str) -> anyhow::Result<String> {
    read_nonempty_env(name).ok_or_else(|| anyhow::anyhow!("missing {name}"))
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::{
        read_nonempty_env, required_nonempty_env, resolve_ai_api_key, resolve_asr_runtime_config,
        resolve_autojoin_suffix, resolve_autojoin_text_channel_id, resolve_config_path,
        resolve_transcription_runtime_config, AiSection, AsrSection, DiscordSection,
        TranscriptionSection,
    };

    struct EnvGuard {
        key: &'static str,
        original: Option<String>,
    }

    impl EnvGuard {
        fn new(key: &'static str) -> Self {
            Self {
                key,
                original: env::var(key).ok(),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(value) = &self.original {
                env::set_var(self.key, value);
            } else {
                env::remove_var(self.key);
            }
        }
    }

    #[test]
    fn environment_helpers_trim_filter_and_resolve_path() {
        let _value_guard = EnvGuard::new("TRANSCRIBE_BOT_TEST_ENV");
        let _path_guard = EnvGuard::new("APP_CONFIG_PATH");
        env::set_var("TRANSCRIBE_BOT_TEST_ENV", "   value   ");
        assert_eq!(
            read_nonempty_env("TRANSCRIBE_BOT_TEST_ENV"),
            Some("value".to_string())
        );

        env::set_var("TRANSCRIBE_BOT_TEST_ENV", "    ");
        assert_eq!(read_nonempty_env("TRANSCRIBE_BOT_TEST_ENV"), None);
        assert!(required_nonempty_env("TRANSCRIBE_BOT_TEST_ENV").is_err());

        env::set_var("TRANSCRIBE_BOT_TEST_ENV", "  token  ");
        assert_eq!(
            required_nonempty_env("TRANSCRIBE_BOT_TEST_ENV").unwrap(),
            "token"
        );

        env::set_var("APP_CONFIG_PATH", "custom.toml");
        assert_eq!(resolve_config_path().to_str(), Some("custom.toml"));

        env::remove_var("APP_CONFIG_PATH");
        assert_eq!(resolve_config_path().to_str(), Some("config.toml"));
    }

    #[test]
    fn api_key_is_only_read_when_its_environment_variable_is_configured() {
        let _default_guard = EnvGuard::new("OPENAI_API_KEY");
        let _explicit_guard = EnvGuard::new("TRANSCRIBE_BOT_TEST_API_KEY");
        env::set_var("OPENAI_API_KEY", "must-not-be-used");
        env::set_var("TRANSCRIBE_BOT_TEST_API_KEY", " explicit-key ");

        let without_key = AiSection {
            model: Some("model-a".to_string()),
            ..AiSection::default()
        };
        assert_eq!(resolve_ai_api_key(Some(&without_key)).unwrap(), None);

        let with_key = AiSection {
            api_key_env: Some("TRANSCRIBE_BOT_TEST_API_KEY".to_string()),
            ..without_key
        };
        assert_eq!(
            resolve_ai_api_key(Some(&with_key)).unwrap().as_deref(),
            Some("explicit-key")
        );
    }

    #[test]
    fn transcription_config_enforces_safety_floors_and_boundaries() {
        let rejected = TranscriptionSection {
            endpoint_silence_ms: Some(10),
            rolling_ingest_max_ms: Some(100),
            rolling_ingest_context_ms: Some(100),
            retention_days: Some(0),
        };
        let cfg = resolve_transcription_runtime_config(Some(&rejected));
        assert_eq!(cfg.endpoint_silence_ms, 400);
        assert_eq!(cfg.rolling_ingest_max_ms, 12_000);
        assert_eq!(cfg.rolling_ingest_context_ms, 1_500);
        assert_eq!(cfg.retention_days, 30);

        let boundaries = TranscriptionSection {
            endpoint_silence_ms: Some(80),
            rolling_ingest_max_ms: Some(4_000),
            rolling_ingest_context_ms: Some(250),
            retention_days: Some(1),
        };
        let cfg = resolve_transcription_runtime_config(Some(&boundaries));
        assert_eq!(cfg.endpoint_silence_ms, 80);
        assert_eq!(cfg.rolling_ingest_max_ms, 4_000);
        assert_eq!(cfg.rolling_ingest_context_ms, 250);
        assert_eq!(cfg.retention_days, 1);
    }

    #[test]
    fn asr_config_requires_model_dir_and_clamps_thread_count() {
        let err = resolve_asr_runtime_config(None, 4).expect_err("missing model dir should fail");
        assert!(err.to_string().contains("config.toml"));

        let high = AsrSection {
            model_dir: Some("models/test".to_string()),
            model_family: None,
            num_threads: Some(99),
        };
        assert_eq!(
            resolve_asr_runtime_config(Some(&high), 4)
                .unwrap()
                .num_threads,
            8
        );

        let invalid = AsrSection {
            num_threads: Some(0),
            ..high
        };
        assert_eq!(
            resolve_asr_runtime_config(Some(&invalid), 2)
                .unwrap()
                .num_threads,
            2
        );
    }

    #[test]
    fn autojoin_suffix_uses_default_for_blank_values() {
        let blank = DiscordSection {
            autojoin_suffix: Some("   ".to_string()),
            ..DiscordSection::default()
        };
        assert_eq!(resolve_autojoin_suffix(Some(&blank)), "[Transcribe]");
    }

    #[test]
    fn autojoin_text_channel_id_is_optional_but_cannot_be_zero() {
        assert_eq!(resolve_autojoin_text_channel_id(None).unwrap(), None);
        assert_eq!(
            resolve_autojoin_text_channel_id(Some(&DiscordSection {
                autojoin_text_channel_id: Some(123),
                ..DiscordSection::default()
            }))
            .unwrap(),
            Some(123)
        );
        assert!(resolve_autojoin_text_channel_id(Some(&DiscordSection {
            autojoin_text_channel_id: Some(0),
            ..DiscordSection::default()
        }))
        .is_err());
    }
}
