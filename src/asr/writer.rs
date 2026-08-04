use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, SystemTime};

use serde::Serialize;
use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, RwLock};

use crate::app::{CallSession, Utterance};

pub async fn transcript_writer_loop(
    session: Arc<RwLock<CallSession>>,
    mut rx: mpsc::Receiver<Utterance>,
    pending_commits: Arc<AtomicUsize>,
    transcript_jsonl_path: PathBuf,
) {
    #[derive(Serialize)]
    struct PersistedUtterance {
        user_id: u64,
        start_offset_ms: u64,
        text: String,
    }

    async fn append_persisted_utterance(path: &Path, item: &PersistedUtterance) {
        let Ok(mut file) = tokio::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
            .await
        else {
            return;
        };
        let Ok(line) = serde_json::to_string(item) else {
            return;
        };
        let _ = file.write_all(format!("{line}\n").as_bytes()).await;
    }

    async fn append_utterance(
        session: &Arc<RwLock<CallSession>>,
        pending_commits: &Arc<AtomicUsize>,
        transcript_jsonl_path: &Path,
        utterance: Utterance,
    ) {
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
        append_persisted_utterance(transcript_jsonl_path, &persisted).await;

        pending_commits.fetch_sub(1, AtomicOrdering::SeqCst);
    }

    while let Some(item) = rx.recv().await {
        append_utterance(
            &session,
            &pending_commits,
            &transcript_jsonl_path,
            item,
        )
        .await;
    }
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
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use super::prune_old_transcripts;

    fn temp_dir(name: &str) -> std::path::PathBuf {
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
        dir
    }

    #[tokio::test]
    async fn prune_old_transcripts_only_deletes_matching_jsonl_files() {
        let dir = temp_dir("prune");
        let keep_non_transcript = dir.join("notes.jsonl");
        let keep_wrong_ext = dir.join("transcript-demo.txt");
        let prune_target = dir.join("transcript-demo.jsonl");

        fs::write(&keep_non_transcript, b"x").expect("write notes");
        fs::write(&keep_wrong_ext, b"x").expect("write txt");
        fs::write(&prune_target, b"x").expect("write transcript");

        let deleted = prune_old_transcripts(&dir, Duration::ZERO).await;
        assert_eq!(deleted, 1);
        assert!(keep_non_transcript.exists());
        assert!(keep_wrong_ext.exists());
        assert!(!prune_target.exists());

        let _ = fs::remove_dir_all(&dir);
    }
}
