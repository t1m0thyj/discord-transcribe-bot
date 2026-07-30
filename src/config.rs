use std::env;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub discord_token: String,
    pub gemini_api_key: String,
    pub gemini_model: String,
    pub asr_streaming_model_dir: String,
    pub asr_offline_model_dir: String,
    pub live_transcript_debug: bool,
    pub enable_denoiser: bool,
    pub autojoin_suffix: String,
}

impl AppConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let discord_token = env::var("DISCORD_TOKEN")?;
        let gemini_api_key = env::var("GEMINI_API_KEY")?;
        let gemini_model = env::var("GEMINI_MODEL")?;
        let asr_streaming_model_dir = env::var("ASR_STREAMING_MODEL_DIR")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "missing ASR_STREAMING_MODEL_DIR: point it at a sherpa-onnx streaming transducer model"
                )
            })?;
        let asr_offline_model_dir = env::var("ASR_OFFLINE_MODEL_DIR")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        let asr_offline_model_dir = asr_offline_model_dir
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "missing ASR_OFFLINE_MODEL_DIR: point it at the authoritative offline finalizer model"
                )
            })?;

        let live_transcript_debug = env::var("LIVE_TRANSCRIPT_DEBUG")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        let enable_denoiser = env::var("ENABLE_DENOISER")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);

        let autojoin_suffix = env::var("AUTOJOIN_SUFFIX")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "[Transcribe]".to_string());

        Ok(Self {
            discord_token,
            gemini_api_key,
            gemini_model,
            asr_streaming_model_dir,
            asr_offline_model_dir,
            live_transcript_debug,
            enable_denoiser,
            autojoin_suffix,
        })
    }
}
