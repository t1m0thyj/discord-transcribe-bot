use serde_json::json;

use anyhow::Context as _;

use super::{AiMessage, AiProviderConfig};

pub(super) fn provider_config(
    api_key: Option<String>,
    model: Option<String>,
) -> anyhow::Result<AiProviderConfig> {
    let api_key = api_key
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .context("missing GEMINI_API_KEY (required when AI_PROVIDER=gemini)")?;
    let model = model
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "gemini-flash-latest".to_string());

    Ok(AiProviderConfig::Gemini { api_key, model })
}

pub async fn generate_chat(
    http: &reqwest::Client,
    api_key: &str,
    model: &str,
    turns: &[AiMessage],
) -> anyhow::Result<String> {
    let contents: Vec<serde_json::Value> = turns
        .iter()
        .map(|m| {
            let role = if m.role.eq_ignore_ascii_case("assistant") {
                "model"
            } else {
                "user"
            };
            json!({
                "role": role,
                "parts": [{ "text": m.text }]
            })
        })
        .collect();

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    );

    let http_resp = http
        .post(&url)
        .json(&json!({ "contents": contents }))
        .send()
        .await?;

    let status = http_resp.status();
    let resp: serde_json::Value = http_resp.json().await?;

    if !status.is_success() {
        let message = resp["error"]["message"]
            .as_str()
            .unwrap_or("unknown Gemini API error");
        return Err(anyhow::anyhow!(
            "Gemini API returned HTTP {}: {}",
            status,
            message
        ));
    }

    if let Some(text) = resp["candidates"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|c| c["content"]["parts"].as_array())
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
    {
        return Ok(text);
    }

    let block_reason = resp["promptFeedback"]["blockReason"].as_str();
    let finish_reason = resp["candidates"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|c| c["finishReason"].as_str());

    Err(anyhow::anyhow!(
        "Gemini returned no text (finish_reason={:?}, block_reason={:?})",
        finish_reason,
        block_reason
    ))
}

#[cfg(test)]
mod tests {
    use super::provider_config;
    use crate::ai::AiProviderConfig;

    #[test]
    fn provider_config_requires_nonempty_api_key() {
        let err = provider_config(Some("   ".to_string()), None).expect_err("missing key should fail");
        assert!(err.to_string().contains("missing GEMINI_API_KEY"));
    }

    #[test]
    fn provider_config_uses_default_model_when_missing() {
        let cfg = provider_config(Some("abc".to_string()), None)
            .expect("valid gemini config");

        match cfg {
            AiProviderConfig::Gemini { api_key, model } => {
                assert_eq!(api_key, "abc");
                assert_eq!(model, "gemini-flash-latest");
            }
            _ => panic!("expected gemini provider"),
        }
    }
}

