use std::collections::VecDeque;

use earshot::Detector;
use nnnoiseless::DenoiseState;
use rubato::{FftFixedInOut, Resampler};

pub const RNNOISE_FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;
const EARSHOT_FRAME_SIZE: usize = 256;
const EARSHOT_VAD_THRESHOLD: f32 = 0.5;
const VAD_HANGOVER_FRAMES: u8 = 16;
const VAD_PREROLL_SAMPLES_16K: usize = 4_800;
const DENOISER_BYPASS_SNR_DB: f32 = 18.0;
const DENOISER_BYPASS_HYSTERESIS_DB: f32 = 3.0;
const HIGHPASS_ALPHA_48K: f32 = 0.991;

pub struct UserDenoiseState {
    denoiser: Box<DenoiseState<'static>>,
    vad: Detector,
    pending: Vec<f32>,
    vad_pending_16k: Vec<f32>,
    warmed_up: bool,
    hp_prev_x: f32,
    hp_prev_y: f32,
    noise_rms_ema: f32,
    last_snr_db: f32,
    resampler_48k_to_16k: FftFixedInOut<f32>,
    resample_pending_48k: Vec<f32>,
    resample_error_count: usize,
    preroll_16k: VecDeque<f32>,
    vad_hangover_frames: u8,
    was_speech_last_tick: bool,
    denoiser_active: bool,
    vad_frame_buf: Vec<f32>,
    resample_in_buf: Vec<f32>,
    resample_out_buf: Vec<f32>,
}

pub struct ProcessedSpeechChunk {
    pub pcm_16k: Vec<f32>,
    pub speech_active: bool,
}

impl UserDenoiseState {
    pub fn new() -> Self {
        Self {
            denoiser: DenoiseState::new(),
            vad: Detector::default(),
            pending: Vec::new(),
            vad_pending_16k: Vec::new(),
            warmed_up: false,
            hp_prev_x: 0.0,
            hp_prev_y: 0.0,
            noise_rms_ema: 0.002,
            last_snr_db: 0.0,
            resampler_48k_to_16k: FftFixedInOut::new(48_000, 16_000, 960, 1)
                .expect("valid fixed 48k->16k resampler config"),
            resample_pending_48k: Vec::new(),
            resample_error_count: 0,
            preroll_16k: VecDeque::with_capacity(VAD_PREROLL_SAMPLES_16K),
            vad_hangover_frames: 0,
            was_speech_last_tick: false,
            denoiser_active: true,
            vad_frame_buf: vec![0.0; EARSHOT_FRAME_SIZE],
            resample_in_buf: Vec::new(),
            resample_out_buf: Vec::new(),
        }
    }

    pub fn push_stereo_pcm(&mut self, input: &[i16], enable_denoiser: bool) -> ProcessedSpeechChunk {
        let mut mono = downmix_stereo_to_mono_unit_scale(input);
        if mono.is_empty() {
            return ProcessedSpeechChunk {
                pcm_16k: Vec::new(),
                speech_active: false,
            };
        }

        self.apply_highpass(&mut mono);

        let rms = compute_rms(&mono);
        self.update_noise_and_snr(rms);
        let use_denoiser = self.select_denoiser_mode(enable_denoiser);
        let cleaned_48k = if use_denoiser {
            self.push_mono_pcm(&mono)
        } else {
            let mut passthrough = self.drain_pending_passthrough();
            passthrough.extend_from_slice(&mono);
            passthrough
        };

        let pcm_16k = self.resample_48k_to_16k_stream(&cleaned_48k);
        let was_speech_last_tick = self.was_speech_last_tick;
        let speech_active = self.passes_vad_earshot(&pcm_16k);
        let mut emitted_pcm_16k = pcm_16k;

        if speech_active {
            if !was_speech_last_tick && !self.preroll_16k.is_empty() {
                let mut with_preroll = Vec::with_capacity(self.preroll_16k.len() + emitted_pcm_16k.len());
                with_preroll.extend(self.preroll_16k.iter().copied());
                with_preroll.extend(emitted_pcm_16k);
                emitted_pcm_16k = with_preroll;
            }
            self.preroll_16k.clear();
        } else {
            self.push_preroll_16k(&emitted_pcm_16k);
        }

        self.was_speech_last_tick = speech_active;

        ProcessedSpeechChunk {
            pcm_16k: emitted_pcm_16k,
            speech_active,
        }
    }

    pub fn push_mono_pcm(&mut self, input: &[f32]) -> Vec<f32> {
        self.pending.extend_from_slice(input);
        let mut out = Vec::new();

        while self.pending.len() >= RNNOISE_FRAME_SIZE {
            let frame: Vec<f32> = self.pending.drain(..RNNOISE_FRAME_SIZE).collect();
            out.extend(self.denoise_frame(&frame));
        }

        out
    }

    pub fn noise_rms_ema(&self) -> f32 {
        self.noise_rms_ema
    }

    fn drain_pending_passthrough(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.pending)
    }

    fn select_denoiser_mode(&mut self, enable_denoiser: bool) -> bool {
        if !enable_denoiser {
            self.denoiser_active = false;
            return false;
        }

        if self.was_speech_last_tick {
            return self.denoiser_active;
        }

        if self.last_snr_db >= DENOISER_BYPASS_SNR_DB + DENOISER_BYPASS_HYSTERESIS_DB {
            self.denoiser_active = false;
        } else if self.last_snr_db <= DENOISER_BYPASS_SNR_DB - DENOISER_BYPASS_HYSTERESIS_DB {
            self.denoiser_active = true;
        }

        self.denoiser_active
    }

    fn update_noise_and_snr(&mut self, rms: f32) {
        let noise_update = if rms <= self.noise_rms_ema * 1.5 { 0.08 } else { 0.005 };
        self.noise_rms_ema = self.noise_rms_ema * (1.0 - noise_update) + rms * noise_update;
        let noise = self.noise_rms_ema.max(1e-4);
        self.last_snr_db = 20.0 * ((rms + 1e-4) / noise).log10();
    }

    fn passes_vad_earshot(&mut self, pcm_16k: &[f32]) -> bool {
        if !pcm_16k.is_empty() {
            self.vad_pending_16k.extend_from_slice(pcm_16k);
        }

        let mut evaluated_any = false;
        let mut speech_active = false;
        while self.vad_pending_16k.len() >= EARSHOT_FRAME_SIZE {
            self.vad_frame_buf
                .copy_from_slice(&self.vad_pending_16k[..EARSHOT_FRAME_SIZE]);
            self.vad_pending_16k.drain(..EARSHOT_FRAME_SIZE);
            let score = self.vad.predict_f32(&self.vad_frame_buf);
            evaluated_any = true;
            if score >= EARSHOT_VAD_THRESHOLD {
                self.vad_hangover_frames = VAD_HANGOVER_FRAMES;
                speech_active = true;
            } else if self.vad_hangover_frames > 0 {
                self.vad_hangover_frames -= 1;
                speech_active = true;
            }
        }

        if evaluated_any {
            return speech_active;
        }

        self.vad_hangover_frames > 0
    }

    fn push_preroll_16k(&mut self, samples: &[f32]) {
        for sample in samples {
            if self.preroll_16k.len() == VAD_PREROLL_SAMPLES_16K {
                self.preroll_16k.pop_front();
            }
            self.preroll_16k.push_back(*sample);
        }
    }

    fn apply_highpass(&mut self, samples: &mut [f32]) {
        for sample in samples {
            let x = *sample;
            let y = HIGHPASS_ALPHA_48K * (self.hp_prev_y + x - self.hp_prev_x);
            self.hp_prev_x = x;
            self.hp_prev_y = y;
            *sample = y;
        }
    }

    fn resample_48k_to_16k_stream(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        self.resample_pending_48k.extend_from_slice(input);

        let in_frames = self.resampler_48k_to_16k.input_frames_next();
        let out_frames = self.resampler_48k_to_16k.output_frames_next();

        let mut out = Vec::with_capacity(
            (self.resample_pending_48k.len() / in_frames)
                .saturating_mul(out_frames),
        );
        self.resample_in_buf.resize(in_frames, 0.0);
        self.resample_out_buf.resize(out_frames, 0.0);

        while self.resample_pending_48k.len() >= in_frames {
            self.resample_in_buf
                .copy_from_slice(&self.resample_pending_48k[..in_frames]);
            self.resample_pending_48k.drain(..in_frames);
            if self
                .resampler_48k_to_16k
                .process_into_buffer(
                    &[self.resample_in_buf.as_slice()],
                    &mut [self.resample_out_buf.as_mut_slice()],
                    None,
                )
                .is_err()
            {
                self.resample_error_count = self.resample_error_count.saturating_add(1);
                continue;
            }
            out.extend_from_slice(&self.resample_out_buf);
        }

        out
    }

    fn denoise_frame(&mut self, input: &[f32]) -> Vec<f32> {
        let mut scaled = [0.0f32; RNNOISE_FRAME_SIZE];
        for (dst, src) in scaled.iter_mut().zip(input.iter().copied()) {
            *dst = src * i16::MAX as f32;
        }

        let mut output = [0.0f32; RNNOISE_FRAME_SIZE];
        self.denoiser.process_frame(&mut output, &scaled);

        if !self.warmed_up {
            self.warmed_up = true;
            return input.to_vec();
        }

        output
            .into_iter()
            .map(|sample| sample / i16::MAX as f32)
            .collect()
    }

    pub fn take_resample_error_count(&mut self) -> usize {
        std::mem::take(&mut self.resample_error_count)
    }
}

impl Default for UserDenoiseState {
    fn default() -> Self {
        Self::new()
    }
}

pub fn downmix_stereo_to_mono_i16_scale(input: &[i16]) -> Vec<f32> {
    let mut out = Vec::with_capacity(input.len() / 2);
    let mut i = 0;
    while i + 1 < input.len() {
        let l = input[i] as f32;
        let r = input[i + 1] as f32;
        out.push((l + r) * 0.5);
        i += 2;
    }
    out
}

pub fn downmix_stereo_to_mono_unit_scale(input: &[i16]) -> Vec<f32> {
    downmix_stereo_to_mono_i16_scale(input)
        .into_iter()
        .map(|v| (v / i16::MAX as f32).clamp(-1.0, 1.0))
        .collect()
}

pub(crate) fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{
        RNNOISE_FRAME_SIZE, UserDenoiseState, compute_rms, downmix_stereo_to_mono_i16_scale,
        downmix_stereo_to_mono_unit_scale,
    };

    #[test]
    fn downmix_i16_scale_averages_stereo_pairs() {
        let input = [100i16, 300, -400, 200];
        let mono = downmix_stereo_to_mono_i16_scale(&input);
        assert_eq!(mono, vec![200.0, -100.0]);
    }

    #[test]
    fn downmix_unit_scale_constrains_to_minus_one_to_one() {
        let input = [i16::MAX, i16::MAX, i16::MIN, i16::MIN];
        let mono = downmix_stereo_to_mono_unit_scale(&input);
        assert!(mono[0] <= 1.0 && mono[0] > 0.99);
        assert!(mono[1] >= -1.0 && mono[1] < -0.99);
    }

    #[test]
    fn compute_rms_handles_empty_and_known_values() {
        assert_eq!(compute_rms(&[]), 0.0);
        let rms = compute_rms(&[1.0, -1.0, 1.0, -1.0]);
        assert!((rms - 1.0).abs() < 1e-6);
    }

    #[test]
    fn push_mono_pcm_buffers_until_full_frame() {
        let mut state = UserDenoiseState::new();
        let almost_frame = vec![0.1f32; RNNOISE_FRAME_SIZE - 10];
        let tail = vec![0.1f32; 10];

        let first_out = state.push_mono_pcm(&almost_frame);
        assert!(first_out.is_empty());

        let second_out = state.push_mono_pcm(&tail);
        assert_eq!(second_out.len(), RNNOISE_FRAME_SIZE);
    }
}
