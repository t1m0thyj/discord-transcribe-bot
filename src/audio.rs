use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serenity::http::Http;
use serenity::all::{GuildId, UserId};
use songbird::events::{Event, EventContext, EventHandler as VoiceEventHandler};
use tokio::sync::mpsc;

use crate::app::Utterance;
use crate::transcription::{
    make_revision_id, transcribe_mono_pcm, AsrEngine, SILENCE_TICKS_THRESHOLD,
    SsrcMap, Streams,
};

const FINALIZE_STABLE_PREVIEW_STREAK: u32 = 2;
const FINALIZE_UNSTABLE_EXTRA_TICKS: u32 = 4;

pub struct SpeakingUpdateHandler {
    pub guild_id: GuildId,
    pub ssrc_to_user: Arc<SsrcMap>,
}

#[serenity::async_trait]
impl VoiceEventHandler for SpeakingUpdateHandler {
    async fn act(&self, ctx: &EventContext<'_>) -> Option<Event> {
        if let EventContext::SpeakingStateUpdate(speaking) = ctx {
            if let Some(user_id) = speaking.user_id {
                self.ssrc_to_user
                    .insert((self.guild_id, speaking.ssrc), UserId::new(user_id.0));
            }
        }
        None
    }
}

pub struct ClientDisconnectHandler {
    pub guild_id: GuildId,
    pub ssrc_to_user: Arc<SsrcMap>,
}

#[serenity::async_trait]
impl VoiceEventHandler for ClientDisconnectHandler {
    async fn act(&self, ctx: &EventContext<'_>) -> Option<Event> {
        let EventContext::ClientDisconnect(disconnect) = ctx else {
            return None;
        };

        let keys: Vec<(GuildId, u32)> = self
            .ssrc_to_user
            .iter()
            .filter_map(|entry| {
                if entry.key().0 == self.guild_id && *entry.value() == UserId::new(disconnect.user_id.0) {
                    Some(*entry.key())
                } else {
                    None
                }
            })
            .collect();

        for key in keys {
            self.ssrc_to_user.remove(&key);
        }

        None
    }
}

pub struct VoiceTickHandler {
    pub http: Arc<Http>,
    pub text_channel: serenity::all::ChannelId,
    pub voice_channel: serenity::all::ChannelId,
    pub started_notified: Arc<AtomicBool>,
    pub guild_id: GuildId,
    pub ssrc_to_user: Arc<SsrcMap>,
    pub streams: Arc<Streams>,
    pub enable_denoiser: bool,
    pub utterance_tx: mpsc::Sender<Utterance>,
    pub transcription_inflight: Arc<AtomicUsize>,
    pub transcript_pending_commits: Arc<AtomicUsize>,
    pub decode_activity: Arc<AtomicUsize>,
    pub decode_failure_activity: Arc<AtomicUsize>,
    pub unmapped_ssrc_activity: Arc<AtomicUsize>,
    pub asr: Arc<AsrEngine>,
    pub asr_finalize: Option<Arc<AsrEngine>>,
    pub live_transcript_debug: bool,
    pub rolling_ingest_max_ms: u64,
    pub rolling_ingest_context_ms: u64,
}

#[serenity::async_trait]
impl VoiceEventHandler for VoiceTickHandler {
    async fn act(&self, ctx: &EventContext<'_>) -> Option<Event> {
        let EventContext::VoiceTick(tick) = ctx else {
            return None;
        };

        let mut currently_speaking = std::collections::HashSet::<UserId>::new();
        let first_preview_samples = 8_000; // 0.5s at 16 kHz
        let max_preview_samples = 128_000; // 8s at 16 kHz
        let max_ingest_samples =
            ((self.rolling_ingest_max_ms as usize) * 16).max(first_preview_samples);
        let base_keep_context_samples = (self.rolling_ingest_context_ms as usize) * 16;

        for (ssrc, data) in &tick.speaking {
            if data.decoded_voice.is_none() && data.packet.is_some() {
                self.decode_failure_activity.fetch_add(1, Ordering::SeqCst);
            }

            let Some(decoded) = &data.decoded_voice else {
                continue;
            };

            let Some(user_id) = self
                .ssrc_to_user
                .get(&(self.guild_id, *ssrc))
                .map(|v| *v)
            else {
                // Decoded audio without an SSRC mapping is not usable yet.
                // Count it as startup receive failure signal so watchdog can recover.
                self.unmapped_ssrc_activity.fetch_add(1, Ordering::SeqCst);
                continue;
            };

            self.decode_activity.fetch_add(1, Ordering::SeqCst);

            let user_key = (self.guild_id, user_id);

            let mut stream = self.streams.entry(user_key).or_default();
            let processed = stream
                .denoiser
                .push_stereo_pcm(decoded, self.enable_denoiser);
            if processed.speech_active {
                currently_speaking.insert(user_id);
            }
            if !processed.speech_active || processed.pcm_16k.is_empty() {
                continue;
            }

            if self
                .started_notified
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                let http = Arc::clone(&self.http);
                let text_channel = self.text_channel;
                let voice_channel = self.voice_channel;
                tokio::spawn(async move {
                    let _ = text_channel
                        .say(&http, format!("Started transcribing in <#{}>.", voice_channel.get()))
                        .await;
                });
            }

            let cleaned = processed.pcm_16k;

            let entry = &mut stream.buffer;
            if entry.utterance_start.is_none() {
                entry.utterance_start = Some(Instant::now());
                entry.current_revision_seq = Some(entry.next_revision_seq);
                entry.next_revision_seq = entry.next_revision_seq.wrapping_add(1);
                entry.last_preview_samples = 0;
                entry.last_preview_text = None;
                entry.frozen_prefix_words = 0;
                entry.stable_preview_streak = 0;
            }
            entry.silent_ticks = 0;
            entry.pcm.extend_from_slice(&cleaned);

            let mut maybe_rollover_final = None;
            if entry.pcm.len() >= max_ingest_samples {
                let max_keep_without_starving = max_ingest_samples.saturating_sub(1_600);
                let keep = base_keep_context_samples
                    .min(max_keep_without_starving)
                    .min(entry.pcm.len());
                let split_at = entry.pcm.len().saturating_sub(keep);
                if split_at >= 1_600 {
                    let start_ts = entry.utterance_start.take().unwrap_or_else(Instant::now);
                    let revision_seq = entry.current_revision_seq.take().unwrap_or_else(|| {
                        let seq = entry.next_revision_seq;
                        entry.next_revision_seq = entry.next_revision_seq.wrapping_add(1);
                        seq
                    });

                    let tail = entry.pcm.split_off(split_at);
                    let chunk = std::mem::replace(&mut entry.pcm, tail);

                    entry.utterance_start = Some(
                        Instant::now()
                            .checked_sub(Duration::from_millis(self.rolling_ingest_context_ms))
                            .unwrap_or_else(Instant::now),
                    );
                    entry.current_revision_seq = Some(entry.next_revision_seq);
                    entry.next_revision_seq = entry.next_revision_seq.wrapping_add(1);
                    entry.last_preview_samples = entry.pcm.len();
                    entry.last_preview_text = None;
                    entry.frozen_prefix_words = 0;
                    entry.stable_preview_streak = 0;

                    maybe_rollover_final = Some((start_ts, revision_seq, chunk));
                }
            }

            let next_preview_samples = if entry.last_preview_samples == 0 {
                first_preview_samples
            } else {
                entry
                    .last_preview_samples
                    .saturating_mul(2)
                    .min(max_preview_samples)
            };

            let maybe_preview = if entry.pcm.len() >= next_preview_samples {
                entry.last_preview_samples = next_preview_samples;
                let start_ts = entry.utterance_start.unwrap_or_else(Instant::now);
                let revision_seq = entry.current_revision_seq.unwrap_or(0);
                Some((start_ts, revision_seq, entry.pcm.clone()))
            } else {
                None
            };

            tracing::trace!(
                guild = %self.guild_id,
                user = %user_id,
                ssrc = %ssrc,
                samples = cleaned.len(),
                "processed 16k pcm tick"
            );

            if let Some((start_ts, revision_seq, pcm)) = maybe_rollover_final {
                let tx = self.utterance_tx.clone();
                let asr = Arc::clone(&self.asr);
                let asr_finalize = self.asr_finalize.as_ref().map(Arc::clone);
                let inflight = Arc::clone(&self.transcription_inflight);
                let pending_commits = Arc::clone(&self.transcript_pending_commits);
                let live_transcript_debug = self.live_transcript_debug;
                self.transcription_inflight.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    struct InflightTaskGuard {
                        counter: Arc<AtomicUsize>,
                    }
                    impl Drop for InflightTaskGuard {
                        fn drop(&mut self) {
                            self.counter.fetch_sub(1, Ordering::SeqCst);
                        }
                    }

                    let _guard = InflightTaskGuard { counter: inflight };

                    if let Some(text) = transcribe_utterance_blocking(&asr, &pcm).await {
                        let revision_id = make_revision_id(user_id, revision_seq);
                        if live_transcript_debug {
                            tracing::debug!(
                                user = %user_id,
                                revision_seq,
                                transcript = %text,
                                "final transcription pass=1 stage=rollover"
                            );
                        }
                        pending_commits.fetch_add(1, Ordering::SeqCst);
                        if tx
                            .send(Utterance {
                                user_id,
                                start_ts,
                                revision_id,
                                is_final: true,
                                text: text.clone(),
                            })
                            .await
                            .is_err()
                        {
                            pending_commits.fetch_sub(1, Ordering::SeqCst);
                            return;
                        }

                        if let Some(final_asr) = asr_finalize {
                            if let Some(refined_text) = transcribe_utterance_blocking(&final_asr, &pcm).await {
                                if should_apply_refinement(&text, &refined_text) {
                                    if live_transcript_debug {
                                        tracing::debug!(
                                            user = %user_id,
                                            revision_seq,
                                            transcript = %refined_text,
                                            "final transcription pass=2 stage=rollover"
                                        );
                                    }
                                    pending_commits.fetch_add(1, Ordering::SeqCst);
                                    if tx
                                        .send(Utterance {
                                            user_id,
                                            start_ts,
                                            revision_id,
                                            is_final: true,
                                            text: refined_text,
                                        })
                                        .await
                                        .is_err()
                                    {
                                        pending_commits.fetch_sub(1, Ordering::SeqCst);
                                    }
                                }
                            }
                        }
                    }
                });
            }

            if let Some((start_ts, revision_seq, pcm)) = maybe_preview {
                let tx = self.utterance_tx.clone();
                let asr = Arc::clone(&self.asr);
                let streams = Arc::clone(&self.streams);
                let inflight = Arc::clone(&self.transcription_inflight);
                let pending_commits = Arc::clone(&self.transcript_pending_commits);
                let live_transcript_debug = self.live_transcript_debug;
                self.transcription_inflight.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    struct InflightTaskGuard {
                        counter: Arc<AtomicUsize>,
                    }
                    impl Drop for InflightTaskGuard {
                        fn drop(&mut self) {
                            self.counter.fetch_sub(1, Ordering::SeqCst);
                        }
                    }

                    let _guard = InflightTaskGuard { counter: inflight };

                    let still_current = streams
                        .get(&user_key)
                        .map(|entry| entry.buffer.current_revision_seq == Some(revision_seq))
                        .unwrap_or(false);
                    if !still_current {
                        return;
                    }

                    if let Some(text) = transcribe_utterance_blocking(&asr, &pcm).await {
                        let Some(refined) = refine_provisional_for_emission(
                            &streams,
                            user_key,
                            revision_seq,
                            text,
                        ) else {
                            return;
                        };

                        if live_transcript_debug {
                            tracing::debug!(
                                user = %user_id,
                                revision_seq,
                                transcript = %refined.text,
                                word_confidence_proxy = %format_word_confidence_proxy(&refined.word_confidence_proxy),
                                "provisional transcription"
                            );
                        }
                        pending_commits.fetch_add(1, Ordering::SeqCst);
                        if tx
                            .send(Utterance {
                                user_id,
                                start_ts,
                                revision_id: make_revision_id(user_id, revision_seq),
                                is_final: false,
                                text: refined.text,
                            })
                            .await
                            .is_ok()
                        {
                        } else {
                            pending_commits.fetch_sub(1, Ordering::SeqCst);
                        }
                    }
                });
            }
        }

        let tracked_users: Vec<(GuildId, UserId)> =
            self.streams.iter().map(|e| *e.key()).collect();
        for user_key in tracked_users {
            if user_key.0 != self.guild_id {
                continue;
            }
            let user_id = user_key.1;
            if currently_speaking.contains(&user_id) {
                continue;
            }

            let mut maybe_job = None;
            if let Some(mut stream) = self.streams.get_mut(&user_key) {
                let flushed = if self.enable_denoiser {
                    stream.denoiser.flush_pending()
                } else {
                    Vec::new()
                };
                let entry = &mut stream.buffer;
                entry.silent_ticks = entry.silent_ticks.saturating_add(1);
                if entry.silent_ticks >= SILENCE_TICKS_THRESHOLD && !entry.pcm.is_empty() {
                    let needs_more_context = entry.last_preview_text.is_some()
                        && entry.stable_preview_streak < FINALIZE_STABLE_PREVIEW_STREAK;
                    let max_silence_wait =
                        SILENCE_TICKS_THRESHOLD.saturating_add(FINALIZE_UNSTABLE_EXTRA_TICKS);
                    if needs_more_context && entry.silent_ticks < max_silence_wait {
                        continue;
                    }

                    entry.pcm.extend(flushed);
                    let start_ts = entry.utterance_start.take().unwrap_or_else(Instant::now);
                    let revision_seq = entry.current_revision_seq.take().unwrap_or_else(|| {
                        let seq = entry.next_revision_seq;
                        entry.next_revision_seq = entry.next_revision_seq.wrapping_add(1);
                        seq
                    });
                    let pcm = std::mem::take(&mut entry.pcm);
                    entry.silent_ticks = 0;
                    entry.last_preview_samples = 0;
                    entry.last_preview_text = None;
                    entry.frozen_prefix_words = 0;
                    entry.stable_preview_streak = 0;
                    maybe_job = Some((start_ts, revision_seq, pcm));
                }
            }

            if let Some((start_ts, revision_seq, pcm)) = maybe_job {
                let tx = self.utterance_tx.clone();
                let asr = Arc::clone(&self.asr);
                let asr_finalize = self.asr_finalize.as_ref().map(Arc::clone);
                let inflight = Arc::clone(&self.transcription_inflight);
                let pending_commits = Arc::clone(&self.transcript_pending_commits);
                let live_transcript_debug = self.live_transcript_debug;
                self.transcription_inflight.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    struct InflightTaskGuard {
                        counter: Arc<AtomicUsize>,
                    }
                    impl Drop for InflightTaskGuard {
                        fn drop(&mut self) {
                            self.counter.fetch_sub(1, Ordering::SeqCst);
                        }
                    }

                    let _guard = InflightTaskGuard { counter: inflight };

                    if let Some(text) = transcribe_utterance_blocking(&asr, &pcm).await {
                        let revision_id = make_revision_id(user_id, revision_seq);
                        if live_transcript_debug {
                            tracing::debug!(
                                user = %user_id,
                                revision_seq,
                                transcript = %text,
                                "final transcription pass=1 stage=silence"
                            );
                        }
                        pending_commits.fetch_add(1, Ordering::SeqCst);
                        if tx
                            .send(Utterance {
                                user_id,
                                start_ts,
                                revision_id,
                                is_final: true,
                                text: text.clone(),
                            })
                            .await
                            .is_ok()
                        {
                            if let Some(final_asr) = asr_finalize {
                                if let Some(refined_text) = transcribe_utterance_blocking(&final_asr, &pcm).await {
                                    if should_apply_refinement(&text, &refined_text) {
                                        if live_transcript_debug {
                                            tracing::debug!(
                                                user = %user_id,
                                                revision_seq,
                                                transcript = %refined_text,
                                                "final transcription pass=2 stage=silence"
                                            );
                                        }
                                        pending_commits.fetch_add(1, Ordering::SeqCst);
                                        if tx
                                            .send(Utterance {
                                                user_id,
                                                start_ts,
                                                revision_id,
                                                is_final: true,
                                                text: refined_text,
                                            })
                                            .await
                                            .is_err()
                                        {
                                            pending_commits.fetch_sub(1, Ordering::SeqCst);
                                        }
                                    }
                                }
                            }
                        } else {
                            pending_commits.fetch_sub(1, Ordering::SeqCst);
                        }
                    }
                });
            }
        }

        None
    }
}

fn refine_provisional_for_emission(
    streams: &Arc<Streams>,
    user_key: (GuildId, UserId),
    revision_seq: u64,
    new_text: String,
) -> Option<RefinedEmission> {
    let mut stream = streams.get_mut(&user_key)?;
    let entry = &mut stream.buffer;
    if entry.current_revision_seq != Some(revision_seq) {
        return None;
    }

    let Some(prev_text) = entry.last_preview_text.clone() else {
        entry.last_preview_text = Some(new_text.clone());
        return Some(RefinedEmission {
            text: new_text.clone(),
            word_confidence_proxy: build_word_confidence_proxy(&new_text, 0),
        });
    };

    let prev_words: Vec<&str> = prev_text.split_whitespace().collect();
    let new_words: Vec<&str> = new_text.split_whitespace().collect();
    let stable_prefix = common_prefix_words(&prev_words, &new_words);
    entry.frozen_prefix_words = entry.frozen_prefix_words.max(stable_prefix);

    let composed = compose_with_frozen_prefix(&prev_words, &new_words, entry.frozen_prefix_words);
    let delta = word_change_delta(&prev_text, &composed);
    if delta <= 1 {
        entry.stable_preview_streak = entry.stable_preview_streak.saturating_add(1);
    } else {
        entry.stable_preview_streak = 0;
    }

    if !is_meaningful_word_change(&prev_text, &composed) {
        return None;
    }

    entry.last_preview_text = Some(composed.clone());
    Some(RefinedEmission {
        text: composed.clone(),
        word_confidence_proxy: build_word_confidence_proxy(&composed, entry.frozen_prefix_words),
    })
}

struct RefinedEmission {
    text: String,
    word_confidence_proxy: Vec<(String, f32)>,
}

fn build_word_confidence_proxy(text: &str, frozen_prefix_words: usize) -> Vec<(String, f32)> {
    let words: Vec<&str> = text.split_whitespace().collect();
    words
        .into_iter()
        .enumerate()
        .map(|(idx, w)| {
            let conf = if idx < frozen_prefix_words { 0.9 } else { 0.45 };
            (w.to_string(), conf)
        })
        .collect()
}

fn format_word_confidence_proxy(items: &[(String, f32)]) -> String {
    items
        .iter()
        .map(|(w, c)| format!("{}:{:.2}", w, c))
        .collect::<Vec<_>>()
        .join(" ")
}

fn common_prefix_words(prev: &[&str], new: &[&str]) -> usize {
    let mut n = 0usize;
    while n < prev.len() && n < new.len() && prev[n].eq_ignore_ascii_case(new[n]) {
        n += 1;
    }
    n
}

fn compose_with_frozen_prefix(prev: &[&str], new: &[&str], frozen_words: usize) -> String {
    let frozen = frozen_words.min(prev.len());
    let mut out = Vec::new();
    out.extend(prev.iter().take(frozen).copied());

    if new.len() > frozen {
        out.extend(new.iter().skip(frozen).copied());
    }

    out.join(" ").trim().to_string()
}

fn is_meaningful_word_change(prev: &str, next: &str) -> bool {
    let delta = word_change_delta(prev, next);
    let prev_norm = prev.split_whitespace().collect::<Vec<_>>().join(" ");
    let next_norm = next.split_whitespace().collect::<Vec<_>>().join(" ");
    if prev_norm.eq_ignore_ascii_case(&next_norm) {
        return false;
    }

    let prev_words: Vec<&str> = prev_norm.split_whitespace().collect();
    if prev_words.len() >= 6 && delta <= 1 {
        return false;
    }

    true
}

fn word_change_delta(prev: &str, next: &str) -> usize {
    let prev_norm = prev.split_whitespace().collect::<Vec<_>>().join(" ");
    let next_norm = next.split_whitespace().collect::<Vec<_>>().join(" ");
    let prev_words: Vec<&str> = prev_norm.split_whitespace().collect();
    let next_words: Vec<&str> = next_norm.split_whitespace().collect();
    let min_len = prev_words.len().min(next_words.len());

    let mut delta = prev_words.len().max(next_words.len()) - min_len;
    for i in 0..min_len {
        if !prev_words[i].eq_ignore_ascii_case(next_words[i]) {
            delta += 1;
        }
    }

    delta
}

async fn transcribe_utterance_blocking(
    asr: &Arc<AsrEngine>,
    pcm_mono: &[f32],
) -> Option<String> {
    transcribe_mono_pcm(Arc::clone(asr), pcm_mono.to_vec()).await
}

fn texts_equivalent(a: &str, b: &str) -> bool {
    a.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .eq_ignore_ascii_case(&b.split_whitespace().collect::<Vec<_>>().join(" "))
}

fn should_apply_refinement(pass1: &str, pass2: &str) -> bool {
    if texts_equivalent(pass1, pass2) {
        return false;
    }

    let w1 = pass1.split_whitespace().count().max(1);
    let w2 = pass2.split_whitespace().count();
    if w2 == 0 {
        return false;
    }

    let ratio = w2 as f32 / w1 as f32;
    if !(0.4..=2.5).contains(&ratio) {
        return false;
    }

    if has_repeated_ngram(pass2, 3, 3) {
        return false;
    }

    true
}

fn has_repeated_ngram(text: &str, n: usize, min_consecutive_repeats: usize) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < n.saturating_mul(min_consecutive_repeats) || n == 0 {
        return false;
    }

    let mut i = 0usize;
    while i + n <= words.len() {
        let candidate = &words[i..i + n];
        let mut repeats = 1usize;
        let mut j = i + n;
        while j + n <= words.len() && words[j..j + n].eq(candidate) {
            repeats += 1;
            if repeats >= min_consecutive_repeats {
                return true;
            }
            j += n;
        }
        i += 1;
    }

    false
}
