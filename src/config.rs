use std::env;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub discord_token: String,
    pub gemini_api_key: String,
    pub gemini_model: String,
    pub asr_model_dir: String,
    pub asr_num_threads: i32,
    pub live_transcript_debug: bool,
    pub enable_denoiser: bool,
    pub endpoint_silence_ms: u64,
    pub rolling_ingest_max_ms: u64,
    pub rolling_ingest_context_ms: u64,
    pub autojoin_suffix: String,
}

impl AppConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let discord_token = env::var("DISCORD_TOKEN")?;
        let gemini_api_key = env::var("GEMINI_API_KEY")?;
        let gemini_model = env::var("GEMINI_MODEL")?;
        let asr_model_dir = env::var("ASR_MODEL_DIR").map_err(|_| {
            anyhow::anyhow!("missing ASR model directory: set ASR_MODEL_DIR")
        })?;
        let detected_cores = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .clamp(1, 8);
        let default_asr_num_threads = if detected_cores >= 4 { 3 } else { detected_cores };
        let asr_num_threads = env::var("ASR_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|v| *v >= 1)
            .unwrap_or(default_asr_num_threads)
            .clamp(1, 8);
        let live_transcript_debug = env::var("LIVE_TRANSCRIPT_DEBUG")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        let enable_denoiser = env::var("ENABLE_DENOISER")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        let endpoint_silence_ms = env::var("ENDPOINT_SILENCE_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v >= 80)
            .unwrap_or(400);

        // Keep in-memory audio bounded for long uninterrupted speech.
        let rolling_ingest_max_ms = env::var("ROLLING_INGEST_MAX_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v >= 4_000)
            .unwrap_or(12_000);

        let rolling_ingest_context_ms = env::var("ROLLING_INGEST_CONTEXT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v >= 250)
            .unwrap_or(1_500);

        let autojoin_suffix = env::var("AUTOJOIN_SUFFIX")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "[Transcribe]".to_string());

        Ok(Self {
            discord_token,
            gemini_api_key,
            gemini_model,
            asr_model_dir,
            asr_num_threads,
            live_transcript_debug,
            enable_denoiser,
            endpoint_silence_ms,
            rolling_ingest_max_ms,
            rolling_ingest_context_ms,
            autojoin_suffix,
        })
    }
}
