use std::future::Future;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde_json::json;

use super::stream::parse_openai_stream;
use super::{truncate_chars, AiMessage, AiProviderConfig};

const TRANSIENT_REQUEST_MAX_RETRIES: usize = 2;
const TRANSIENT_REQUEST_BACKOFF_BASE: Duration = Duration::from_millis(500);
const TRANSIENT_REQUEST_MAX_RETRY_AFTER: Duration = Duration::from_secs(30);

pub(super) async fn check_model_available(
    http: &reqwest::Client,
    provider: &AiProviderConfig,
) -> anyhow::Result<()> {
    let url = format!("{}/models", provider.base_url.trim_end_matches('/'));
    let response = send_with_retry("openai-compatible", &provider.model, true, || {
        let request = http.get(&url);
        let request = match provider.api_key.as_deref() {
            Some(api_key) => request.bearer_auth(api_key),
            None => request,
        };
        request.send()
    })
    .await
    .context("failed to request OpenAI-compatible API model list")?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read OpenAI-compatible API model list")?;
    ensure_success_response(status, &body)?;
    ensure_model_is_listed(&body, &provider.model)
}

pub(super) async fn generate_chat(
    http: &reqwest::Client,
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
    turns: &[AiMessage],
) -> anyhow::Result<String> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
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
    let payload = json!({ "model": model, "messages": messages, "stream": true });

    let mut response = send_with_retry("openai-compatible", model, false, || {
        let request = http.post(&url).json(&payload);
        let request = match api_key {
            Some(api_key) => request.bearer_auth(api_key),
            None => request,
        };
        request.send()
    })
    .await?;

    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .await
            .context("failed to read OpenAI-compatible API error response")?;
        ensure_success_response(status, &body)?;
        unreachable!("a non-success status always returns an error")
    }

    parse_openai_stream(&mut response).await
}

async fn send_with_retry<F, Fut>(
    provider: &'static str,
    model: &str,
    retry_timeouts: bool,
    mut send: F,
) -> Result<reqwest::Response, reqwest::Error>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<reqwest::Response, reqwest::Error>>,
{
    for retry in 0..=TRANSIENT_REQUEST_MAX_RETRIES {
        match send().await {
            Ok(response) => {
                if !is_retryable_status(response.status()) || retry == TRANSIENT_REQUEST_MAX_RETRIES
                {
                    return Ok(response);
                }

                let status = response.status();
                let delay = retry_delay(response.headers(), retry);
                tracing::info!(
                    provider,
                    model,
                    status = %status,
                    retry_attempt = retry + 1,
                    retry_delay_ms = delay.as_millis(),
                    "retrying transient AI API response"
                );
                tokio::time::sleep(delay).await;
            }
            Err(error) => {
                // Generation requests retry connection failures but not timeouts, which may have
                // reached the provider. Idempotent checks may opt into timeout retries.
                if !(error.is_connect() || (retry_timeouts && error.is_timeout()))
                    || retry == TRANSIENT_REQUEST_MAX_RETRIES
                {
                    return Err(error);
                }

                let delay = retry_delay(&HeaderMap::new(), retry);
                tracing::info!(
                    provider,
                    model,
                    error = %error,
                    retry_attempt = retry + 1,
                    retry_delay_ms = delay.as_millis(),
                    "retrying failed AI API request"
                );
                tokio::time::sleep(delay).await;
            }
        }
    }

    unreachable!("the retry loop always returns after the final attempt")
}

fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    matches!(
        status,
        reqwest::StatusCode::TOO_MANY_REQUESTS
            | reqwest::StatusCode::INTERNAL_SERVER_ERROR
            | reqwest::StatusCode::BAD_GATEWAY
            | reqwest::StatusCode::SERVICE_UNAVAILABLE
            | reqwest::StatusCode::GATEWAY_TIMEOUT
    )
}

fn retry_delay(headers: &HeaderMap, retry: usize) -> Duration {
    let exponential_delay = TRANSIENT_REQUEST_BACKOFF_BASE.saturating_mul(1_u32 << retry);
    if let Some(retry_after) = headers
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs)
        .map(|delay| delay.min(TRANSIENT_REQUEST_MAX_RETRY_AFTER))
    {
        return retry_after;
    }

    let jitter_limit_ms = (exponential_delay.as_millis() / 4) as u64;
    let jitter_ms = if jitter_limit_ms == 0 {
        0
    } else {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64
            % (jitter_limit_ms + 1)
    };
    exponential_delay.saturating_add(Duration::from_millis(jitter_ms))
}

fn ensure_success_response(status: reqwest::StatusCode, body: &str) -> anyhow::Result<()> {
    if status.is_success() {
        return Ok(());
    }

    let response: serde_json::Value = serde_json::from_str(body).unwrap_or(serde_json::Value::Null);
    let message = extract_error_message(&response)
        .map(ToOwned::to_owned)
        .or_else(|| {
            let body = body.trim();
            (!body.is_empty()).then(|| truncate_chars(body, 500))
        })
        .unwrap_or_else(|| "unknown API error".to_string());
    anyhow::bail!(
        "OpenAI-compatible API returned HTTP {}: {}",
        status,
        message
    )
}

fn extract_error_message(response: &serde_json::Value) -> Option<&str> {
    response["error"]["message"]
        .as_str()
        .or_else(|| response["error"].as_str())
        .or_else(|| response["message"].as_str())
}

fn ensure_model_is_listed(body: &str, model: &str) -> anyhow::Result<()> {
    let response: serde_json::Value = serde_json::from_str(body)
        .context("OpenAI-compatible API returned invalid JSON while listing models")?;
    let models = response["data"]
        .as_array()
        .context("OpenAI-compatible API returned a model list without a data array")?;
    if models
        .iter()
        .any(|candidate| candidate["id"].as_str() == Some(model))
    {
        return Ok(());
    }

    anyhow::bail!("OpenAI-compatible API model list does not contain configured model {model:?}")
}

#[cfg(test)]
mod tests {
    use std::io::{Read as _, Write as _};
    use std::net::TcpListener;
    use std::thread;

    use reqwest::header::{HeaderValue, RETRY_AFTER};

    use super::*;
    use crate::ai::{AiClient, AiProviderConfig};

    fn serve_http_responses(responses: Vec<String>) -> (String, thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test HTTP server");
        let address = listener.local_addr().expect("test server address");
        let handle = thread::spawn(move || {
            responses
                .into_iter()
                .map(|response| {
                    let (mut stream, _) = listener.accept().expect("accept test request");
                    stream
                        .set_read_timeout(Some(Duration::from_secs(5)))
                        .expect("set test read timeout");
                    let mut request = Vec::new();
                    let mut buffer = [0_u8; 4096];
                    loop {
                        let read = stream.read(&mut buffer).expect("read test request");
                        if read == 0 {
                            break;
                        }
                        request.extend_from_slice(&buffer[..read]);
                        if request_is_complete(&request) {
                            break;
                        }
                    }
                    stream
                        .write_all(response.as_bytes())
                        .expect("write test response");
                    String::from_utf8(request).expect("UTF-8 test request")
                })
                .collect()
        });
        (format!("http://{address}/v1"), handle)
    }

    fn request_is_complete(request: &[u8]) -> bool {
        let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
            return false;
        };
        let body_start = header_end + 4;
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .and_then(|value| value.trim().parse::<usize>().ok())
            })
            .unwrap_or(0);
        request.len() >= body_start + content_length
    }

    fn test_response(status: &str, content_type: &str, body: &str, extra_headers: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{extra_headers}Connection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn init_test_crypto_provider() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }

    #[test]
    fn retry_policy_only_retries_transient_statuses() {
        assert!(is_retryable_status(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(is_retryable_status(
            reqwest::StatusCode::SERVICE_UNAVAILABLE
        ));
        assert!(!is_retryable_status(reqwest::StatusCode::BAD_REQUEST));
        assert!(!is_retryable_status(reqwest::StatusCode::UNAUTHORIZED));
    }

    #[test]
    fn retry_delay_is_exponential_and_honors_bounded_retry_after() {
        let headers = HeaderMap::new();
        let first = retry_delay(&headers, 0);
        assert!(first >= TRANSIENT_REQUEST_BACKOFF_BASE);
        assert!(first <= TRANSIENT_REQUEST_BACKOFF_BASE.saturating_mul(5) / 4);

        let second_base = TRANSIENT_REQUEST_BACKOFF_BASE.saturating_mul(2);
        let second = retry_delay(&headers, 1);
        assert!(second >= second_base);
        assert!(second <= second_base.saturating_mul(5) / 4);

        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("120"));
        assert_eq!(retry_delay(&headers, 0), TRANSIENT_REQUEST_MAX_RETRY_AFTER);
    }

    #[test]
    fn http_error_parser_surfaces_provider_message() {
        let error = ensure_success_response(
            reqwest::StatusCode::SERVICE_UNAVAILABLE,
            r#"{"error":{"message":"provider unavailable"}}"#,
        )
        .expect_err("HTTP error should fail");
        assert!(error.to_string().contains("provider unavailable"));
    }

    #[test]
    fn model_list_check_requires_the_configured_model() {
        ensure_model_is_listed(r#"{"data":[{"id":"model-a"}]}"#, "model-a")
            .expect("configured model should be listed");

        let error = ensure_model_is_listed(r#"{"data":[{"id":"model-a"}]}"#, "model-b")
            .expect_err("unlisted model should fail");
        assert!(error.to_string().contains("model-b"));
    }

    #[tokio::test]
    async fn generation_streams_and_sends_an_explicit_api_key() {
        init_test_crypto_provider();
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let response = test_response("200 OK", "text/event-stream", body, "");
        let (base_url, server) = serve_http_responses(vec![response]);
        let client = AiClient::new(
            AiProviderConfig {
                base_url,
                api_key: Some("secret-key".to_string()),
                model: "model-a".to_string(),
            },
            5,
        )
        .unwrap();

        assert_eq!(
            client.ask("transcript", "question", None).await.unwrap(),
            "hello"
        );
        let requests = server.join().expect("test server should finish");
        let request = &requests[0];
        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(request
            .to_ascii_lowercase()
            .contains("authorization: bearer secret-key\r\n"));
        assert!(request.contains("\"stream\":true"));
    }

    #[tokio::test]
    async fn model_check_retries_transient_errors_without_implicit_authentication() {
        init_test_crypto_provider();
        let unavailable = test_response(
            "503 Service Unavailable",
            "application/json",
            r#"{"error":{"message":"try again"}}"#,
            "Retry-After: 0\r\n",
        );
        let listed = test_response(
            "200 OK",
            "application/json",
            r#"{"data":[{"id":"model-a"}]}"#,
            "",
        );
        let (base_url, server) = serve_http_responses(vec![unavailable, listed]);
        let client = AiClient::new(
            AiProviderConfig {
                base_url,
                api_key: None,
                model: "model-a".to_string(),
            },
            5,
        )
        .unwrap();

        client.check_model_available().await.unwrap();
        let requests = server.join().expect("test server should finish");
        assert_eq!(requests.len(), 2);
        for request in requests {
            assert!(request.starts_with("GET /v1/models HTTP/1.1"));
            assert!(!request.to_ascii_lowercase().contains("authorization:"));
        }
    }
}
