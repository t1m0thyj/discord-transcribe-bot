use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use serenity::all::{GuildId, UserId};

use crate::app::{GuildRuntime, Utterance};

use super::transcription::{transcribe_mono_pcm, AsrEngine};

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
        let mut queue = self
            .queue
            .lock()
            .expect("decode queue mutex poisoned");
        let dropped = if queue.len() >= self.capacity {
            queue.pop_front()
        } else {
            None
        };
        queue.push_back(job);
        drop(queue);
        self.notify.notify_one();
        dropped
    }
}

pub fn decode_queue_depth() -> usize {
    let dispatcher = DecodeDispatcher::global();
    let queue = dispatcher
        .queue
        .lock()
        .expect("decode queue mutex poisoned");
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
                    let mut queue = dispatcher
                        .queue
                        .lock()
                        .expect("decode queue mutex poisoned");
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
    let decode_result = transcribe_utterance_blocking(&job.asr, job.pcm).await;
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
    job.runtime
        .decode_last_ms
        .store(decode_ms, Ordering::SeqCst);
    job.runtime
        .decode_queue_wait_last_ms
        .store(queue_wait_ms, Ordering::SeqCst);

    if let Some(text) = decode_result {
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
        }
    }

    job.runtime
        .transcription_inflight
        .fetch_sub(1, Ordering::SeqCst);
}

async fn transcribe_utterance_blocking(
    asr: &Arc<AsrEngine>,
    pcm_mono: Vec<f32>,
) -> Option<String> {
    transcribe_mono_pcm(Arc::clone(asr), pcm_mono).await
}
