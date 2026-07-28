use serde_json::json;

pub async fn ask_gemini(
    api_key: &str,
    model: &str,
    transcript_context: &str,
    question: &str,
    prior_turns: Option<&[(String, String)]>,
) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    let mut contents = vec![json!({
        "role": "user",
        "parts": [{ "text": format!(
            "You are answering questions about a meeting transcript. Transcript:\n\n{transcript_context}"
        )}]
    })];

    if let Some(turns) = prior_turns {
        for (role, text) in turns {
            contents.push(json!({ "role": role, "parts": [{"text": text}] }));
        }
    }

    contents.push(json!({ "role": "user", "parts": [{"text": question}] }));

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    );

    let http_resp = client
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
        .filter(|s| !s.trim().is_empty())
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
