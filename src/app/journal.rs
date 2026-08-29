use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use serenity::all::UserId;
use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, RwLock};

use super::{CallSession, GuildRuntime, Utterance};

#[derive(Deserialize, Serialize)]
struct PersistedUtterance {
    user_id: u64,
    start_offset_ms: u64,
    text: String,
}

pub async fn transcript_writer_loop(
    session: Arc<RwLock<CallSession>>,
    mut rx: mpsc::Receiver<Utterance>,
    runtime: Arc<GuildRuntime>,
    transcript_jsonl_path: PathBuf,
) -> Result<()> {
    async fn append_persisted_utterance(path: &Path, item: &PersistedUtterance) -> Result<()> {
        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
            .await
            .with_context(|| format!("failed to open transcript journal {}", path.display()))?;
        let line =
            serde_json::to_string(item).context("failed to serialize transcript utterance")?;
        file.write_all(format!("{line}\n").as_bytes())
            .await
            .with_context(|| format!("failed to append transcript journal {}", path.display()))?;
        file.flush()
            .await
            .with_context(|| format!("failed to flush transcript journal {}", path.display()))?;
        Ok(())
    }

    async fn append_utterance(
        session: &Arc<RwLock<CallSession>>,
        transcript_jsonl_path: &Path,
        utterance: Utterance,
    ) -> Result<()> {
        let mut lock = session.write().await;

        let start_offset_ms = utterance
            .start_ts
            .saturating_duration_since(lock.started_mono)
            .as_millis() as u64;
        lock.transcript.push(utterance.clone());

        let persisted = PersistedUtterance {
            user_id: utterance.user_id.get(),
            start_offset_ms,
            text: utterance.text,
        };

        drop(lock);
        append_persisted_utterance(transcript_jsonl_path, &persisted).await
    }

    let mut first_error = None;
    while let Some(item) = rx.recv().await {
        let result = append_utterance(&session, &transcript_jsonl_path, item).await;
        runtime
            .transcript_pending_commits
            .fetch_sub(1, AtomicOrdering::SeqCst);
        if let Err(error) = result {
            tracing::error!("failed to persist transcript utterance: {error:#}");
            if first_error.is_none() {
                first_error = Some(error);
            }
        }
    }

    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

pub(super) async fn load_persisted_transcript(
    path: &Path,
    started_mono: std::time::Instant,
) -> Vec<Utterance> {
    let Ok(content) = tokio::fs::read_to_string(path).await else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for line in content.lines() {
        let Ok(item) = serde_json::from_str::<PersistedUtterance>(line) else {
            continue;
        };
        out.push(Utterance {
            user_id: UserId::new(item.user_id),
            start_ts: started_mono + Duration::from_millis(item.start_offset_ms),
            text: item.text,
        });
    }
    out.sort_by_key(|utterance| utterance.start_ts);
    out
}

pub async fn prune_old_transcripts(dir: &Path, retention: Duration) -> usize {
    let Some(cutoff) = SystemTime::now().checked_sub(retention) else {
        return 0;
    };

    let mut deleted = 0usize;
    let mut entries = match tokio::fs::read_dir(dir).await {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(dir = %dir.display(), "failed to scan transcript directory for cleanup: {e:#}");
            return 0;
        }
    };

    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("transcript-") {
            continue;
        }

        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !matches!(ext, "jsonl") {
            continue;
        }

        let Ok(metadata) = entry.metadata().await else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        if modified > cutoff {
            continue;
        }

        match tokio::fs::remove_file(&path).await {
            Ok(()) => {
                deleted = deleted.saturating_add(1);
            }
            Err(e) => {
                tracing::warn!(file = %path.display(), "failed to remove old transcript file: {e:#}");
            }
        }
    }

    deleted
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use chrono::Utc;
    use serenity::all::{ChannelId, UserId};
    use tokio::sync::{mpsc, RwLock};

    use super::{load_persisted_transcript, prune_old_transcripts, transcript_writer_loop};
    use crate::app::{CallSession, GuildRuntime, Utterance};

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn temp_dir(name: &str) -> TempDir {
        let unique = format!(
            "transcribe-bot-tests-{}-{}-{}",
            name,
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("valid clock")
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        fs::create_dir_all(&dir).expect("create temp dir");
        TempDir(dir)
    }

    #[tokio::test]
    async fn prune_old_transcripts_only_deletes_matching_jsonl_files() {
        let dir = temp_dir("prune");
        let keep_non_transcript = dir.path().join("notes.jsonl");
        let keep_wrong_ext = dir.path().join("transcript-demo.txt");
        let prune_target = dir.path().join("transcript-demo.jsonl");

        fs::write(&keep_non_transcript, b"x").expect("write notes");
        fs::write(&keep_wrong_ext, b"x").expect("write txt");
        fs::write(&prune_target, b"x").expect("write transcript");

        let deleted = prune_old_transcripts(dir.path(), Duration::ZERO).await;
        assert_eq!(deleted, 1);
        assert!(keep_non_transcript.exists());
        assert!(keep_wrong_ext.exists());
        assert!(!prune_target.exists());
    }

    #[tokio::test]
    async fn journal_round_trips_through_writer_and_loader() {
        let dir = temp_dir("journal-roundtrip");
        let path = dir.path().join("transcript.jsonl");
        let started_mono = Instant::now();
        let session = Arc::new(RwLock::new(CallSession {
            voice_channel: ChannelId::new(1),
            text_channel: ChannelId::new(2),
            transcript: Vec::new(),
            transcript_jsonl_path: path.clone(),
            started_at: Utc::now(),
            started_mono,
        }));
        let (runtime_tx, _runtime_rx) = mpsc::channel(1);
        let runtime = Arc::new(GuildRuntime::new(runtime_tx));
        let (tx, rx) = mpsc::channel(8);
        let writer = tokio::spawn(transcript_writer_loop(
            Arc::clone(&session),
            rx,
            Arc::clone(&runtime),
            path.clone(),
        ));

        for (offset_ms, text) in [(2_000, "second \"quoted\""), (500, "first\nline")] {
            runtime
                .transcript_pending_commits
                .fetch_add(1, Ordering::SeqCst);
            tx.send(Utterance {
                user_id: UserId::new(42),
                start_ts: started_mono + Duration::from_millis(offset_ms),
                text: text.to_string(),
            })
            .await
            .expect("enqueue utterance");
        }
        drop(tx);
        writer
            .await
            .expect("writer task completes")
            .expect("writer persists all utterances");

        assert_eq!(runtime.transcript_pending_commits.load(Ordering::SeqCst), 0);
        let loaded = load_persisted_transcript(&path, started_mono).await;
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].text, "first\nline");
        assert_eq!(loaded[1].text, "second \"quoted\"");
    }

    #[tokio::test]
    async fn journal_loader_skips_malformed_lines_and_saturates_negative_offsets() {
        let dir = temp_dir("journal-malformed");
        let path = dir.path().join("transcript.jsonl");
        fs::write(
            &path,
            "{\"user_id\":7,\"start_offset_ms\":0,\"text\":\"first\"}\nnot-json\n{\"user_id\":8,\"start_offset_ms\":20,\"text\":\"second\"}\n",
        )
        .expect("write journal");

        let started_mono = Instant::now();
        let loaded = load_persisted_transcript(&path, started_mono).await;
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].user_id, UserId::new(7));
        assert_eq!(loaded[0].start_ts, started_mono);
        assert_eq!(loaded[1].user_id, UserId::new(8));
    }
}
