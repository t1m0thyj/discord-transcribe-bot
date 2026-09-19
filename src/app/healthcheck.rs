use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const RETENTION: Duration = Duration::from_secs(120);
const MAX_SSRCS: usize = 64;

#[derive(Clone, Copy, Default)]
pub struct SourceCounts {
    pub packets: u64,
    pub decoded_packets: u64,
    pub mapped_packets: u64,
    pub concealed_frames: u64,
    /// A fresh voice-gateway speaking indication with no usable mapped audio.
    pub speaking_without_mapped_audio_since: Option<Instant>,
}

#[derive(Clone, Default)]
pub struct HealthSnapshot(pub HashMap<u32, SourceCounts>);

struct SourceState {
    counts: SourceCounts,
    last_seen: Instant,
    last_packet: Option<Instant>,
}

#[derive(Default)]
pub struct ReceiveHealth {
    sources: Mutex<HashMap<u32, SourceState>>,
    driver_disconnected: AtomicBool,
}

impl ReceiveHealth {
    pub fn packet(&self, ssrc: u32) {
        self.update(ssrc, Instant::now(), |source, now| {
            source.counts.packets += 1;
            source.last_packet = Some(now);
        });
    }

    pub fn decoded(&self, ssrc: u32, mapped: bool, packet_backed: bool) {
        self.update(ssrc, Instant::now(), |source, _| {
            let counts = &mut source.counts;
            if packet_backed {
                counts.decoded_packets += 1;
                if mapped {
                    counts.mapped_packets += 1;
                    counts.speaking_without_mapped_audio_since = None;
                }
            } else {
                counts.concealed_frames += 1;
            }
        });
    }

    pub fn speaking(&self, ssrc: u32, microphone: bool) {
        self.speaking_at(ssrc, microphone, Instant::now());
    }

    fn speaking_at(&self, ssrc: u32, microphone: bool, now: Instant) {
        self.update(ssrc, now, |source, now| {
            if !microphone {
                source.counts.speaking_without_mapped_audio_since = None;
            } else if source
                .last_packet
                .is_none_or(|last| now.saturating_duration_since(last) > Duration::from_secs(1))
            {
                source.counts.speaking_without_mapped_audio_since = Some(now);
            }
        });
    }

    pub fn forget(&self, ssrc: u32) {
        self.sources
            .lock()
            .expect("receive health mutex poisoned")
            .remove(&ssrc);
    }

    pub fn driver_disconnected(&self) -> bool {
        self.driver_disconnected.load(Ordering::SeqCst)
    }

    pub fn set_driver_disconnected(&self, disconnected: bool) {
        self.driver_disconnected
            .store(disconnected, Ordering::SeqCst);
    }

    fn update(&self, ssrc: u32, now: Instant, action: impl FnOnce(&mut SourceState, Instant)) {
        let mut sources = self.sources.lock().expect("receive health mutex poisoned");
        if !sources.contains_key(&ssrc) && sources.len() >= MAX_SSRCS {
            sources.retain(|_, source| now.duration_since(source.last_seen) < RETENTION);
            if sources.len() >= MAX_SSRCS {
                if let Some(oldest) = sources
                    .iter()
                    .min_by_key(|(_, source)| source.last_seen)
                    .map(|(ssrc, _)| *ssrc)
                {
                    sources.remove(&oldest);
                }
            }
        }
        let source = sources.entry(ssrc).or_insert_with(|| SourceState {
            counts: SourceCounts::default(),
            last_seen: now,
            last_packet: None,
        });
        source.last_seen = now;
        action(source, now);
    }

    pub fn snapshot(&self) -> HealthSnapshot {
        let sources = self.sources.lock().expect("receive health mutex poisoned");
        HealthSnapshot(
            sources
                .iter()
                .map(|(ssrc, source)| (*ssrc, source.counts))
                .collect(),
        )
    }

    pub fn last_packet_at(&self) -> Option<Instant> {
        self.sources
            .lock()
            .expect("receive health mutex poisoned")
            .values()
            .filter_map(|source| source.last_packet)
            .max()
    }

    pub fn clear(&self) {
        self.sources
            .lock()
            .expect("receive health mutex poisoned")
            .clear();
        self.driver_disconnected.store(false, Ordering::SeqCst);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceProblem {
    Decode,
    Mapping,
    NoMappedAudioAfterSpeaking,
}

pub fn assess(
    previous: &HealthSnapshot,
    current: &HealthSnapshot,
    now: Instant,
) -> Vec<(u32, SourceProblem)> {
    let mut problems = Vec::new();
    for (&ssrc, counts) in &current.0 {
        let old = previous.0.get(&ssrc).copied().unwrap_or_default();
        let packets = counts.packets.saturating_sub(old.packets);
        let decoded = counts.decoded_packets.saturating_sub(old.decoded_packets);
        let mapped = counts.mapped_packets.saturating_sub(old.mapped_packets);
        // Eight packets (~160 ms at 20-ms pacing) across the rolling window
        // catch brief or intermittent speech; two watchdog polls must agree.
        if packets >= 8 && decoded.saturating_mul(2) < packets {
            problems.push((ssrc, SourceProblem::Decode));
        } else if decoded >= 8 && mapped.saturating_mul(2) < decoded {
            problems.push((ssrc, SourceProblem::Mapping));
        } else if counts
            .speaking_without_mapped_audio_since
            .is_some_and(|since| {
                (Duration::from_secs(5)..=Duration::from_secs(30))
                    .contains(&now.saturating_duration_since(since))
            })
        {
            problems.push((ssrc, SourceProblem::NoMappedAudioAfterSpeaking));
        }
    }
    problems
}

pub fn mapped_audio_observed(snapshot: &HealthSnapshot, ssrc: u32, minimum_packets: u64) -> bool {
    snapshot
        .0
        .get(&ssrc)
        .is_some_and(|counts| counts.mapped_packets >= minimum_packets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_sources_are_not_failed() {
        assert!(assess(
            &HealthSnapshot::default(),
            &HealthSnapshot::default(),
            Instant::now()
        )
        .is_empty());
    }

    #[test]
    fn a_good_speaker_does_not_hide_another_source() {
        let health = ReceiveHealth::default();
        for _ in 0..8 {
            health.packet(1);
            health.decoded(1, true, true);
            health.packet(2);
        }
        assert_eq!(
            assess(
                &HealthSnapshot::default(),
                &health.snapshot(),
                Instant::now()
            ),
            vec![(2, SourceProblem::Decode)]
        );
    }

    #[test]
    fn concealment_is_not_success_and_mapping_is_per_source() {
        let health = ReceiveHealth::default();
        for _ in 0..8 {
            health.packet(1);
            health.decoded(1, false, true);
            health.packet(2);
            health.decoded(2, true, false);
        }
        let problems = assess(
            &HealthSnapshot::default(),
            &health.snapshot(),
            Instant::now(),
        );
        assert!(problems.contains(&(1, SourceProblem::Mapping)));
        assert!(problems.contains(&(2, SourceProblem::Decode)));
    }

    #[test]
    fn gateway_speaking_without_any_rtp_is_evidence_after_grace() {
        let health = ReceiveHealth::default();
        let now = Instant::now();
        health.speaking_at(42, true, now - Duration::from_secs(7));
        assert_eq!(
            assess(&HealthSnapshot::default(), &health.snapshot(), now),
            vec![(42, SourceProblem::NoMappedAudioAfterSpeaking)]
        );
        health.packet(42);
        assert_eq!(
            assess(&HealthSnapshot::default(), &health.snapshot(), now),
            vec![(42, SourceProblem::NoMappedAudioAfterSpeaking)]
        );
        health.decoded(42, true, true);
        assert!(assess(&HealthSnapshot::default(), &health.snapshot(), now).is_empty());
    }

    #[test]
    fn muted_or_stale_speaking_does_not_trigger_recovery() {
        let health = ReceiveHealth::default();
        let now = Instant::now();
        health.speaking_at(42, false, now - Duration::from_secs(7));
        assert!(assess(&HealthSnapshot::default(), &health.snapshot(), now).is_empty());
        health.speaking_at(42, true, now - Duration::from_secs(40));
        assert!(assess(&HealthSnapshot::default(), &health.snapshot(), now).is_empty());
    }

    #[test]
    fn repeated_short_bursts_accumulate_across_the_window() {
        let health = ReceiveHealth::default();
        let before = health.snapshot();
        for _ in 0..4 {
            health.packet(2);
        }
        assert!(assess(&before, &health.snapshot(), Instant::now()).is_empty());
        for _ in 0..4 {
            health.packet(2);
        }
        assert_eq!(
            assess(&before, &health.snapshot(), Instant::now()),
            vec![(2, SourceProblem::Decode)]
        );
    }

    #[test]
    fn leave_and_rejoin_discard_old_receive_evidence() {
        let health = ReceiveHealth::default();
        let now = Instant::now();
        health.speaking_at(42, true, now - Duration::from_secs(7));
        health.forget(42);
        assert!(assess(&HealthSnapshot::default(), &health.snapshot(), now).is_empty());

        for _ in 0..8 {
            health.packet(42);
        }
        health.set_driver_disconnected(true);
        health.clear();
        assert!(!health.driver_disconnected());
        assert!(assess(&HealthSnapshot::default(), &health.snapshot(), now).is_empty());
    }

    #[test]
    fn successful_packet_backed_audio_clears_gateway_failure() {
        let health = ReceiveHealth::default();
        let now = Instant::now();
        health.speaking_at(42, true, now - Duration::from_secs(7));
        health.packet(42);
        health.decoded(42, true, true);
        assert!(assess(&HealthSnapshot::default(), &health.snapshot(), now).is_empty());
    }

    #[test]
    fn a_different_speaker_cannot_confirm_recovery() {
        let health = ReceiveHealth::default();
        for _ in 0..3 {
            health.packet(1);
            health.decoded(1, true, true);
        }
        let snapshot = health.snapshot();
        assert!(mapped_audio_observed(&snapshot, 1, 3));
        assert!(!mapped_audio_observed(&snapshot, 2, 3));
    }
}
