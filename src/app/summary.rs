use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use serenity::all::UserId;
use serenity::prelude::Context;

use super::{AppState, Utterance};

const THREAD_SUMMARY_MAX_CHARS: usize = 1_800;

pub(super) async fn format_transcript(
    ctx: &Context,
    transcript: &[Utterance],
    started_at: chrono::DateTime<chrono::Utc>,
) -> String {
    let by_user = resolve_display_names(ctx, transcript).await;
    let mut lines = build_transcript_lines(transcript, &by_user);

    lines.insert(0, String::new());
    lines.insert(0, "## Transcript".to_string());
    lines.insert(0, String::new());
    lines.insert(
        0,
        format!(
            "**Started:** {} UTC",
            started_at.format("%Y-%m-%d %H:%M:%S")
        ),
    );
    lines.insert(0, String::new());
    lines.insert(0, "# Meeting Transcript".to_string());

    lines.join("\n")
}

pub(super) async fn format_export_markdown(
    ctx: &Context,
    transcript: &[Utterance],
    started_at: chrono::DateTime<chrono::Utc>,
    call_duration: Duration,
    title: &str,
    summary: Option<&str>,
    include_summary_in_markdown: bool,
) -> String {
    let by_user = resolve_display_names(ctx, transcript).await;
    format_export_markdown_with_names(
        transcript,
        &by_user,
        started_at,
        call_duration,
        title,
        summary,
        include_summary_in_markdown,
    )
}

fn format_export_markdown_with_names(
    transcript: &[Utterance],
    by_user: &HashMap<UserId, String>,
    started_at: chrono::DateTime<chrono::Utc>,
    call_duration: Duration,
    title: &str,
    summary: Option<&str>,
    include_summary_in_markdown: bool,
) -> String {
    let attendees = attendees_in_order(transcript, &by_user);
    let duration = format_duration(call_duration);

    let mut out = Vec::new();
    out.push("---".to_string());
    out.push(format!("title: \"{}\"", yaml_escape_double_quoted(title)));
    out.push("type: meeting".to_string());
    out.push(format!(
        "date: {}",
        started_at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
    ));
    out.push(format!("duration: \"{}\"", duration));
    out.push("source: discord".to_string());
    out.push("status: complete".to_string());
    if !attendees.is_empty() {
        out.push("attendees:".to_string());
        for attendee in attendees {
            out.push(format!("  - {}", yaml_single_line_scalar(&attendee)));
        }
    }
    out.push("---".to_string());
    out.push(String::new());
    if include_summary_in_markdown {
        if let Some(summary_text) = summary.map(str::trim).filter(|s| !s.is_empty()) {
            out.push(summary_text.to_string());
            out.push(String::new());
        }
    }
    out.push("## Transcript".to_string());
    out.push(String::new());
    out.extend(build_transcript_lines(transcript, &by_user));
    out.push(String::new());

    out.join("\n")
}

pub(super) fn format_call_title(started_at: chrono::DateTime<chrono::Utc>) -> String {
    format!("Transcript {}", started_at.format("%Y-%m-%d %H:%M:%S UTC"))
}

pub(super) async fn maybe_generate_post_call_summary(
    ctx: &Context,
    state: &Arc<AppState>,
    transcript: &[Utterance],
    started_at: chrono::DateTime<chrono::Utc>,
) -> Option<String> {
    if !state.post_call_summary_enabled {
        return None;
    }

    if transcript.is_empty() {
        return None;
    }

    let transcript_context = format_transcript(ctx, transcript, started_at).await;
    let timeout = Duration::from_secs(state.post_call_summary_timeout_secs.max(5));

    match tokio::time::timeout(timeout, state.ai.summarize_transcript(&transcript_context)).await {
        Ok(Ok(summary)) => {
            let summary = summary.trim().to_string();
            if summary.is_empty() {
                tracing::warn!("auto-summary returned empty text");
                None
            } else {
                Some(summary)
            }
        }
        Ok(Err(e)) => {
            tracing::warn!("auto-summary failed: {e:#}");
            None
        }
        Err(_) => {
            tracing::warn!(timeout_secs = state.post_call_summary_timeout_secs, "auto-summary timed out");
            None
        }
    }
}

pub(super) fn format_summary_thread_message(summary: &str) -> String {
    let trimmed = summary.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut message = trimmed.to_string();
    if message.chars().count() > THREAD_SUMMARY_MAX_CHARS {
        let keep = THREAD_SUMMARY_MAX_CHARS.saturating_sub(18);
        let truncated: String = message.chars().take(keep).collect();
        message = format!("{truncated}\n\n(truncated)");
    }
    message
}

async fn resolve_display_names(ctx: &Context, transcript: &[Utterance]) -> HashMap<UserId, String> {
    let mut by_user = HashMap::<UserId, String>::new();
    for utt in transcript {
        if by_user.contains_key(&utt.user_id) {
            continue;
        }
        let name = match utt.user_id.to_user(&ctx.http).await {
            Ok(u) => u.display_name().to_string(),
            Err(_) => format!("{}", utt.user_id.get()),
        };
        by_user.insert(utt.user_id, name);
    }
    by_user
}

fn attendees_in_order(transcript: &[Utterance], by_user: &HashMap<UserId, String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut attendees = Vec::new();
    for utt in transcript {
        if !seen.insert(utt.user_id) {
            continue;
        }
        if let Some(name) = by_user.get(&utt.user_id) {
            attendees.push(name.clone());
        }
    }
    attendees
}

fn build_transcript_lines(transcript: &[Utterance], by_user: &HashMap<UserId, String>) -> Vec<String> {
    if transcript.is_empty() {
        return vec!["_No captured speech._".to_string()];
    }

    let mut lines = Vec::with_capacity(transcript.len());
    let first = transcript[0].start_ts;

    for utt in transcript {
        let display = by_user
            .get(&utt.user_id)
            .cloned()
            .unwrap_or_else(|| format!("{}", utt.user_id.get()));

        let delta = utt.start_ts.saturating_duration_since(first);
        let stamp = format_transcript_stamp(delta);
        lines.push(format!("[{display} {stamp}] {}", utt.text));
    }

    lines
}

fn format_transcript_stamp(delta: Duration) -> String {
    let total = delta.as_secs();
    let mm = total / 60;
    let ss = total % 60;
    format!("{mm}:{ss:02}")
}

fn format_duration(duration: Duration) -> String {
    let total = duration.as_secs();
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;

    if hours > 0 {
        format!("{hours}h {minutes}m {seconds}s")
    } else {
        format!("{minutes}m {seconds}s")
    }
}

fn yaml_escape_double_quoted(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\r', "\\r")
        .replace('\n', "\\n")
}

fn yaml_single_line_scalar(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '-' | '_' | '.'))
    {
        value.to_string()
    } else {
        format!("\"{}\"", yaml_escape_double_quoted(value))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;
    use std::time::Instant;

    use serenity::all::UserId;

    use super::{
        build_transcript_lines, format_call_title, format_duration, format_export_markdown_with_names,
        format_summary_thread_message, format_transcript_stamp, yaml_escape_double_quoted,
        yaml_single_line_scalar,
    };
    use crate::app::Utterance;

    #[test]
    fn transcript_stamp_formats_minutes_seconds() {
        assert_eq!(format_transcript_stamp(Duration::from_secs(65)), "1:05");
    }

    #[test]
    fn transcript_lines_use_display_names_timestamps_and_id_fallbacks() {
        let started = Instant::now();
        let transcript = vec![
            Utterance {
                user_id: UserId::new(123),
                start_ts: started,
                text: "Hello".to_string(),
            },
            Utterance {
                user_id: UserId::new(456),
                start_ts: started + Duration::from_secs(65),
                text: "Hi there".to_string(),
            },
        ];
        let names = HashMap::from([(UserId::new(123), "Alice".to_string())]);

        assert_eq!(
            build_transcript_lines(&transcript, &names),
            vec!["[Alice 0:00] Hello", "[456 1:05] Hi there"]
        );
        assert_eq!(
            build_transcript_lines(&[], &names),
            vec!["_No captured speech._"]
        );
    }

    #[test]
    fn export_markdown_escapes_hostile_attendees_inside_frontmatter() {
        let started_mono = Instant::now();
        let transcript = vec![Utterance {
            user_id: UserId::new(123),
            start_ts: started_mono,
            text: "Hello".to_string(),
        }];
        let names = HashMap::from([(UserId::new(123), "Eve\n---\nstatus: draft".to_string())]);
        let started_at = chrono::DateTime::parse_from_rfc3339("2026-08-04T12:34:56Z")
            .expect("valid timestamp")
            .with_timezone(&chrono::Utc);

        let markdown = format_export_markdown_with_names(
            &transcript,
            &names,
            started_at,
            Duration::from_secs(65),
            "Call",
            Some("Summary"),
            true,
        );

        assert_eq!(markdown.matches("\n---\n").count(), 2);
        assert!(markdown.contains("  - \"Eve\\n---\\nstatus: draft\""));
        assert!(markdown.contains("Summary\n\n## Transcript"));
        assert!(!markdown.contains("\nstatus: draft\n"));
    }

    #[test]
    fn duration_formats_with_or_without_hours() {
        assert_eq!(format_duration(Duration::from_secs(125)), "2m 5s");
        assert_eq!(format_duration(Duration::from_secs(3_665)), "1h 1m 5s");
    }

    #[test]
    fn summary_thread_message_truncates_long_content() {
        let long = "a".repeat(2_100);
        let message = format_summary_thread_message(&long);
        assert!(message.ends_with("\n\n(truncated)"));
        assert!(message.chars().count() <= 1_800);
    }

    #[test]
    fn yaml_helpers_escape_and_quote_when_needed() {
        assert_eq!(yaml_escape_double_quoted("a\\b\"c"), "a\\\\b\\\"c");
        assert_eq!(yaml_single_line_scalar("Alice-1"), "Alice-1");
        assert_eq!(yaml_single_line_scalar("Alice:1"), "\"Alice:1\"");
        assert_eq!(
            yaml_single_line_scalar("Alice\n---\ntype: evil"),
            "\"Alice\\n---\\ntype: evil\""
        );
    }

    #[test]
    fn call_title_has_stable_full_format() {
        let started = chrono::DateTime::parse_from_rfc3339("2026-08-04T12:34:56Z")
            .expect("valid timestamp")
            .with_timezone(&chrono::Utc);
        let title = format_call_title(started);
        assert_eq!(title, "Transcript 2026-08-04 12:34:56 UTC");
    }
}
