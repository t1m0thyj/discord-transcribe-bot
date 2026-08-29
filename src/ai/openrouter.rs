use anyhow::Context as _;
use serde_json::json;

use super::{send_with_retry, AiMessage, AiProviderConfig};

const DEFAULT_MODEL: &str = "openrouter/free";
const CHAT_COMPLETIONS_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

pub(super) fn provider_config(
    api_key: Option<String>,
    model: Option<String>,
) -> anyhow::Result<AiProviderConfig> {
    let api_key = api_key
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .context("missing OPENROUTER_API_KEY (required when AI_PROVIDER=openrouter)")?;
    let model = model
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    Ok(AiProviderConfig::OpenRouter { api_key, model })
}

pub async fn generate_chat(
    http: &reqwest::Client,
    api_key: &str,
    model: &str,
    turns: &[AiMessage],
) -> anyhow::Result<String> {
    let messages: Vec<serde_json::Value> = turns
        .iter()
        .map(|message| {
            let role = if message.role.eq_ignore_ascii_case("assistant") {
                "assistant"
            } else {
                "user"
            };
            json!({ "role": role, "content": message.text })
        })
        .collect();

    let payload = json!({ "model": model, "messages": messages });
    let http_resp = send_with_retry("openrouter", model, || {
        http.post(CHAT_COMPLETIONS_URL)
            .bearer_auth(api_key)
            .header("X-OpenRouter-Title", "discord-live-transcribe")
            .json(&payload)
            .send()
    })
    .await?;

    let status = http_resp.status();
    let response: serde_json::Value = http_resp.json().await?;
    parse_openrouter_response(status, &response)
}

fn parse_openrouter_response(
    status: reqwest::StatusCode,
    response: &serde_json::Value,
) -> anyhow::Result<String> {
    if !status.is_success() {
        let message = response["error"]["message"]
            .as_str()
            .or_else(|| response["message"].as_str())
            .unwrap_or("unknown OpenRouter API error");
        return Err(anyhow::anyhow!(
            "OpenRouter API returned HTTP {}: {}",
            status,
            message
        ));
    }

    response["choices"]
        .as_array()
        .and_then(|choices| choices.first())
        .and_then(|choice| choice["message"]["content"].as_str())
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow::anyhow!("OpenRouter returned no text"))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{parse_openrouter_response, provider_config, DEFAULT_MODEL};
    use crate::ai::AiProviderConfig;

    #[test]
    fn provider_config_requires_nonempty_api_key() {
        let error =
            provider_config(Some("  ".to_string()), None).expect_err("missing API key should fail");
        assert!(error.to_string().contains("missing OPENROUTER_API_KEY"));
    }

    #[test]
    fn provider_config_uses_default_model_when_missing() {
        let config =
            provider_config(Some("key".to_string()), None).expect("valid OpenRouter configuration");

        match config {
            AiProviderConfig::OpenRouter { api_key, model } => {
                assert_eq!(api_key, "key");
                assert_eq!(model, DEFAULT_MODEL);
            }
            _ => panic!("expected OpenRouter provider"),
        }
    }

    #[test]
    fn parser_returns_trimmed_content() {
        let response = json!({ "choices": [{ "message": { "content": "  hello  " } }] });
        assert_eq!(
            parse_openrouter_response(reqwest::StatusCode::OK, &response).unwrap(),
            "hello"
        );
    }

    #[test]
    fn parser_surfaces_api_errors_and_rejects_empty_content() {
        let error = parse_openrouter_response(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            &json!({ "error": { "message": "rate limited" } }),
        )
        .expect_err("HTTP error should fail");
        assert!(error.to_string().contains("rate limited"));

        let error = parse_openrouter_response(reqwest::StatusCode::OK, &json!({ "choices": [] }))
            .expect_err("empty response should fail");
        assert!(error.to_string().contains("returned no text"));
    }
}
