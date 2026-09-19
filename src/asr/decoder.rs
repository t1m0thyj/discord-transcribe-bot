use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use serenity::all::{GuildId, UserId};

use crate::app::{GuildRuntime, Utterance};

use super::pipeline::{transcribe_mono_pcm, AsrEngine};

const DECODE_QUEUE_CAPACITY: usize = 8;

pub(super) struct DecodeJob {
    pub guild_id: GuildId,
    pub user_id: UserId,
    pub start_ts: Instant,
    pub stage: &'static str,
    pub pcm: Vec<f32>,
    pub enqueued_at: Instant,
    pub runtime: Arc<GuildRuntime>,
    pub asr: Arc<AsrEngine>,
    pub live_transcript_debug: bool,
}

struct DecodeDispatcher {
    queue: Mutex<VecDeque<DecodeJob>>,
    notify: tokio::sync::Notify,
    capacity: usize,
}

impl DecodeDispatcher {
    fn global() -> Arc<Self> {
        static INSTANCE: OnceLock<Arc<DecodeDispatcher>> = OnceLock::new();
        Arc::clone(INSTANCE.get_or_init(|| {
            let dispatcher = Arc::new(DecodeDispatcher {
                queue: Mutex::new(VecDeque::new()),
                notify: tokio::sync::Notify::new(),
                capacity: DECODE_QUEUE_CAPACITY,
            });
            spawn_decode_worker(Arc::clone(&dispatcher));
            dispatcher
        }))
    }

    fn enqueue(&self, job: DecodeJob) -> Option<DecodeJob> {
        let mut queue = self.lock_queue();
        let dropped = push_bounded(&mut queue, job, self.capacity);
        drop(queue);
        self.notify.notify_one();
        dropped
    }

    fn lock_queue(&self) -> MutexGuard<'_, VecDeque<DecodeJob>> {
        self.queue.lock().expect("decode queue mutex poisoned")
    }
}

fn push_bounded<T>(queue: &mut VecDeque<T>, item: T, capacity: usize) -> Option<T> {
    debug_assert!(capacity > 0);
    let dropped = if queue.len() >= capacity {
        queue.pop_front()
    } else {
        None
    };
    queue.push_back(item);
    dropped
}

pub fn decode_queue_depth() -> usize {
    let dispatcher = DecodeDispatcher::global();
    let queue = dispatcher.lock_queue();
    queue.len()
}

pub fn decode_queue_capacity() -> usize {
    DECODE_QUEUE_CAPACITY
}

pub(super) fn queue_decode_job(job: DecodeJob) {
    let dispatcher = DecodeDispatcher::global();
    if let Some(dropped) = dispatcher.enqueue(job) {
        dropped
            .runtime
            .decode_shed_total
            .fetch_add(1, Ordering::SeqCst);
        dropped
            .runtime
            .transcription_inflight
            .fetch_sub(1, Ordering::SeqCst);
        tracing::warn!(
            guild = %dropped.guild_id,
            user = %dropped.user_id,
            "decode queue full; dropped oldest queued chunk"
        );
    }
}

fn spawn_decode_worker(dispatcher: Arc<DecodeDispatcher>) {
    tokio::spawn(async move {
        loop {
            let job = loop {
                let maybe_job = {
                    let mut queue = dispatcher.lock_queue();
                    queue.pop_front()
                };

                if let Some(job) = maybe_job {
                    break job;
                }

                dispatcher.notify.notified().await;
            };

            process_decode_job(job).await;
        }
    });
}

async fn process_decode_job(job: DecodeJob) {
    let queue_wait_ms = job.enqueued_at.elapsed().as_millis() as usize;
    let audio_ms = (job.pcm.len().saturating_mul(1000)) / 16_000;
    let decode_started = Instant::now();
    let decode_result = transcribe_mono_pcm(Arc::clone(&job.asr), job.pcm).await;
    let decode_ms = decode_started.elapsed().as_millis() as usize;

    job.runtime.decode_jobs_total.fetch_add(1, Ordering::SeqCst);
    job.runtime
        .decode_audio_total_ms
        .fetch_add(audio_ms, Ordering::SeqCst);
    job.runtime
        .decode_total_ms
        .fetch_add(decode_ms, Ordering::SeqCst);
    job.runtime
        .decode_queue_wait_total_ms
        .fetch_add(queue_wait_ms, Ordering::SeqCst);
    if let Some(text) = match decode_result {
        Ok(text) => text,
        Err(error) => {
            job.runtime
                .asr_decode_error_total
                .fetch_add(1, Ordering::SeqCst);
            tracing::warn!(
                guild = %job.guild_id,
                user = %job.user_id,
                stage = job.stage,
                "ASR decode failed; continuing with later chunks: {error:#}"
            );
            None
        }
    } {
        if job.live_transcript_debug {
            tracing::debug!(
                user = %job.user_id,
                transcript = %text,
                stage = job.stage,
                "final transcription"
            );
        }

        job.runtime
            .decode_jobs_with_text
            .fetch_add(1, Ordering::SeqCst);

        job.runtime
            .transcript_pending_commits
            .fetch_add(1, Ordering::SeqCst);
        if job
            .runtime
            .utterance_tx
            .send(Utterance {
                user_id: job.user_id,
                start_ts: job.start_ts,
                text,
            })
            .await
            .is_err()
        {
            job.runtime
                .transcript_pending_commits
                .fetch_sub(1, Ordering::SeqCst);
            tracing::warn!(
                guild = %job.guild_id,
                user = %job.user_id,
                "journal writer unavailable; dropped decoded utterance"
            );
        }
    }

    job.runtime
        .transcription_inflight
        .fetch_sub(1, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::push_bounded;

    #[test]
    fn bounded_queue_keeps_fifo_order_until_full() {
        let mut queue = VecDeque::new();
        assert_eq!(push_bounded(&mut queue, 1, 2), None);
        assert_eq!(push_bounded(&mut queue, 2, 2), None);
        assert_eq!(queue.into_iter().collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn bounded_queue_sheds_oldest_item_when_full() {
        let mut queue = VecDeque::from([1, 2]);
        assert_eq!(push_bounded(&mut queue, 3, 2), Some(1));
        assert_eq!(queue.into_iter().collect::<Vec<_>>(), vec![2, 3]);
    }
}
