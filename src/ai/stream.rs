use anyhow::Context as _;

use super::truncate_chars;

pub(super) async fn parse_openai_stream(
    response: &mut reqwest::Response,
) -> anyhow::Result<String> {
    let mut stream = ChatStreamAccumulator::default();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("failed while reading OpenAI-compatible API stream")?
    {
        stream.push(&chunk)?;
    }
    stream.finish()
}

#[derive(Default)]
struct ChatStreamAccumulator {
    pending: Vec<u8>,
    event_data: Vec<String>,
    output: String,
    saw_done: bool,
    finish_reason: Option<String>,
    block_reason: Option<String>,
    saw_refusal: bool,
    saw_reasoning: bool,
    saw_tool_calls: bool,
}

impl ChatStreamAccumulator {
    fn push(&mut self, bytes: &[u8]) -> anyhow::Result<()> {
        self.pending.extend_from_slice(bytes);
        while let Some(newline) = self.pending.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=newline).collect();
            self.process_line(&line[..line.len() - 1])?;
        }
        Ok(())
    }

    fn process_line(&mut self, line: &[u8]) -> anyhow::Result<()> {
        let line = std::str::from_utf8(line)
            .context("OpenAI-compatible API stream contained non-UTF-8 data")?
            .trim_end_matches('\r');
        if line.is_empty() {
            return self.process_event();
        }
        if line.starts_with(':') {
            return Ok(());
        }
        if let Some(data) = line.strip_prefix("data:") {
            self.event_data
                .push(data.strip_prefix(' ').unwrap_or(data).to_string());
        }
        Ok(())
    }

    fn process_event(&mut self) -> anyhow::Result<()> {
        if self.event_data.is_empty() {
            return Ok(());
        }

        let data = self.event_data.join("\n");
        self.event_data.clear();
        if data.trim() == "[DONE]" {
            self.saw_done = true;
            return Ok(());
        }

        let event: serde_json::Value = serde_json::from_str(&data)
            .context("OpenAI-compatible API returned an invalid JSON stream event")?;
        if let Some(message) = extract_error_message(&event) {
            anyhow::bail!(
                "OpenAI-compatible API stream returned an error: {}",
                truncate_chars(message, 500)
            )
        }

        if let Some(reason) = event["prompt_feedback"]["block_reason"]
            .as_str()
            .or_else(|| event["promptFeedback"]["blockReason"].as_str())
        {
            self.block_reason = Some(truncate_chars(reason, 100));
        }

        if let Some(choice) = event["choices"]
            .as_array()
            .and_then(|choices| choices.first())
        {
            if let Some(reason) = choice["finish_reason"].as_str() {
                self.finish_reason = Some(truncate_chars(reason, 100));
            }
            let delta = &choice["delta"];
            append_text_content(&mut self.output, &delta["content"]);
            if self.output.is_empty() {
                append_text_content(&mut self.output, &choice["message"]["content"]);
            }
            self.saw_refusal |=
                !delta["refusal"].is_null() || !choice["message"]["refusal"].is_null();
            self.saw_reasoning |= !delta["reasoning_content"].is_null()
                || !choice["message"]["reasoning_content"].is_null();
            self.saw_tool_calls |=
                !delta["tool_calls"].is_null() || !choice["message"]["tool_calls"].is_null();
        }

        Ok(())
    }

    fn finish(mut self) -> anyhow::Result<String> {
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            self.process_line(&pending)?;
        }
        self.process_event()?;

        if !self.saw_done {
            anyhow::bail!(
                "OpenAI-compatible API stream ended before [DONE]{}",
                self.diagnostic_suffix()
            )
        }

        let output = self.output.trim().to_string();
        if output.is_empty() {
            anyhow::bail!(
                "OpenAI-compatible API returned no text{}",
                self.diagnostic_suffix()
            )
        }
        Ok(output)
    }

    fn diagnostic_suffix(&self) -> String {
        let mut details = Vec::new();
        if let Some(reason) = &self.finish_reason {
            details.push(format!("finish_reason={reason}"));
        }
        if let Some(reason) = &self.block_reason {
            details.push(format!("block_reason={reason}"));
        }
        if self.saw_refusal {
            details.push("refusal=true".to_string());
        }
        if self.saw_reasoning {
            details.push("reasoning_without_text=true".to_string());
        }
        if self.saw_tool_calls {
            details.push("tool_calls=true".to_string());
        }
        if details.is_empty() {
            String::new()
        } else {
            format!(" ({})", details.join(", "))
        }
    }
}

fn extract_error_message(response: &serde_json::Value) -> Option<&str> {
    response["error"]["message"]
        .as_str()
        .or_else(|| response["error"].as_str())
        .or_else(|| response["message"].as_str())
}

fn append_text_content(output: &mut String, content: &serde_json::Value) {
    if let Some(text) = content.as_str() {
        output.push_str(text);
        return;
    }
    if let Some(parts) = content.as_array() {
        for part in parts {
            if let Some(text) = part["text"].as_str() {
                output.push_str(text);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ChatStreamAccumulator;

    #[test]
    fn handles_chunk_boundaries_heartbeats_and_done() {
        let mut stream = ChatStreamAccumulator::default();
        stream
            .push(b": keepalive\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"hel")
            .unwrap();
        stream
            .push(b"lo \"},\"finish_reason\":null}]}\r\n\r\n")
            .unwrap();
        stream
            .push(b"data: {\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
            .unwrap();

        assert_eq!(stream.finish().unwrap(), "hello world");
    }

    #[test]
    fn reports_no_text_diagnostics() {
        let mut stream = ChatStreamAccumulator::default();
        stream
            .push(
                b"data: {\"choices\":[{\"delta\":{\"refusal\":\"blocked\"},\"finish_reason\":\"content_filter\"}]}\n\ndata: [DONE]\n\n",
            )
            .unwrap();
        let error = stream
            .finish()
            .expect_err("refusal should not produce text");
        let message = error.to_string();
        assert!(message.contains("finish_reason=content_filter"));
        assert!(message.contains("refusal=true"));
    }

    #[test]
    fn surfaces_provider_errors_and_incomplete_streams() {
        let mut failed = ChatStreamAccumulator::default();
        let error = failed
            .push(b"data: {\"error\":{\"message\":\"provider unavailable\"}}\n\n")
            .expect_err("provider stream error should fail");
        assert!(error.to_string().contains("provider unavailable"));

        let mut incomplete = ChatStreamAccumulator::default();
        incomplete
            .push(b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")
            .unwrap();
        let error = incomplete
            .finish()
            .expect_err("missing done marker should fail");
        assert!(error.to_string().contains("before [DONE]"));
    }
}
