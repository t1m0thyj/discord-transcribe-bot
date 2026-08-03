use serde_json::json;

use anyhow::Context as _;

use super::{AiMessage, AiProviderConfig};

pub(super) fn provider_config(
    model: Option<String>,
    base_url: Option<String>,
) -> anyhow::Result<AiProviderConfig> {
    let model = model
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .context("missing OLLAMA_MODEL (required when AI_PROVIDER=ollama)")?;
    Ok(AiProviderConfig::Ollama {
        base_url: resolve_ollama_base_url(base_url),
        model,
    })
}

pub async fn generate_chat(
    http: &reqwest::Client,
    base_url: &str,
    model: &str,
    turns: &[AiMessage],
) -> anyhow::Result<String> {
    let url = format!("{}/api/chat", base_url.trim_end_matches('/'));

    let messages: Vec<serde_json::Value> = turns
        .iter()
        .map(|m| {
            let role = if m.role.eq_ignore_ascii_case("assistant") {
                "assistant"
            } else {
                "user"
            };
            json!({
                "role": role,
                "content": m.text
            })
        })
        .collect();

    let http_resp = http
        .post(&url)
        .json(&json!({
            "model": model,
            "stream": false,
            "messages": messages
        }))
        .send()
        .await?;

    let status = http_resp.status();
    let resp: serde_json::Value = http_resp.json().await?;

    if !status.is_success() {
        let message = resp["error"]
            .as_str()
            .or_else(|| resp["message"].as_str())
            .unwrap_or("unknown Ollama API error");
        return Err(anyhow::anyhow!(
            "Ollama API returned HTTP {}: {}",
            status,
            message
        ));
    }

    let text = resp["message"]["content"]
        .as_str()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow::anyhow!("Ollama returned no text"))?;

    Ok(text)
}

fn resolve_ollama_base_url(base_url: Option<String>) -> String {
    if let Some(base_url) = base_url
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
    {
        return base_url.trim_end_matches('/').to_string();
    }

    "http://127.0.0.1:11434".to_string()
}
