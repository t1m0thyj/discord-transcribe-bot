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

#[cfg(test)]
mod tests {
    use super::{provider_config, resolve_ollama_base_url};
    use crate::ai::AiProviderConfig;

    #[test]
    fn resolve_ollama_base_url_defaults_when_missing() {
        assert_eq!(
            resolve_ollama_base_url(None),
            "http://127.0.0.1:11434".to_string()
        );
    }

    #[test]
    fn resolve_ollama_base_url_trims_whitespace_and_trailing_slash() {
        assert_eq!(
            resolve_ollama_base_url(Some(" http://localhost:11434/  ".to_string())),
            "http://localhost:11434".to_string()
        );
    }

    #[test]
    fn provider_config_requires_model() {
        let err = provider_config(Some("  ".to_string()), None)
            .expect_err("missing model should fail");
        assert!(err.to_string().contains("missing OLLAMA_MODEL"));
    }

    #[test]
    fn provider_config_applies_base_url_normalization() {
        let cfg = provider_config(
            Some("llama3".to_string()),
            Some("http://localhost:11434/".to_string()),
        )
        .expect("valid ollama config");

        match cfg {
            AiProviderConfig::Ollama { base_url, model } => {
                assert_eq!(base_url, "http://localhost:11434");
                assert_eq!(model, "llama3");
            }
            _ => panic!("expected ollama provider"),
        }
    }
}
