# Architecture

Short, current design summary for the Discord live transcription bot.

## Stack

- Rust + tokio
- serenity + songbird receive mode
- sherpa-onnx local ASR
- optional nnnoiseless denoise
- Gemini via reqwest for Q&A

## Runtime flow

1. /join starts a guild-scoped call session.
2. Voice events populate guild-scoped speaker maps and per-user stream state (denoiser + audio buffer in a single map entry).
3. Audio pipeline performs downmix, high-pass filter, optional RNNoise denoise with SNR hysteresis and speech-latched mode switching, speech-gated AGC, anti-aliased 48 kHz to 16 kHz resampling (rubato FFT), and earshot VAD gating with a 300 ms pre-roll ring, before single-model ASR segmentation/finalization on silence.
4. Rolling ingest bounds memory for long speech: old chunk is finalized at a low-RMS cut point near the rollover boundary, recent context tail is retained, and transcript commit logic trims strong tail/head word overlap for final utterances from the same speaker.
5. Finalized utterances are queued and merged by a transcript writer with a small reorder window, and appended incrementally to a per-session JSONL file so transcripts survive a crash or a failed Discord upload.
6. /ask and /log read committed transcript state for the active call.
7. /leave (or empty channel auto-finalize) performs settle+flush, disconnects, settle+flush again, merges the persisted JSONL, writes the transcript to local disk, then uploads it and creates a Q&A thread. Drain loops are bounded by timeouts.
8. Thread messages use transcript + in-memory thread history; transcript can be lazily restored from the starter attachment after restart.

## Commands

- /join
- /leave
- /status
- /log
- /ask
- /autojoin

Suffix-based auto-join is implemented: channels ending with the configured marker (default ` [Transcribe]`) are auto-joined when a non-bot user enters.

## Key state

- Utterance: user_id, start_ts, text
- UserAudioBuffer: audio PCM + segmentation tracking
- CallSession: voice_channel, text_channel, transcript, started_at, started_mono
- ThreadContext: transcript + turn history
- AppState: active calls, thread contexts, ASR engine, guild-scoped buffers/maps, drain counters

## Ordering and consistency

- ASR work runs concurrently (spawn + spawn_blocking).
- Transcript writes are append-only and ordered by start timestamp with a watermark.
- Finalization uses inflight/pending counters and explicit flush/drain waits.

## Modules

- src/main.rs: bootstrap, intents, event routing
- src/config.rs: env config
- src/audio.rs: voice ingest, buffering, segmentation, ASR dispatch
- src/transcription.rs: ASR setup, resampling, transcript writer
- src/app/mod.rs: app-level types/state and routing facade
- src/app/commands.rs: slash command handlers and session start path
- src/app/session.rs: lifecycle finalize/recovery/watchdog/export/thread context
- src/gemini.rs: Gemini request/response

## Config knobs

- DISCORD_TOKEN
- GEMINI_API_KEY
- GEMINI_MODEL
- ASR_MODEL_DIR
- ASR_MODEL_FAMILY (optional)
- LIVE_TRANSCRIPT_DEBUG
- ENABLE_DENOISER
- ROLLING_INGEST_MAX_MS
- ROLLING_INGEST_CONTEXT_MS
- AUTOJOIN_SUFFIX (optional; default `[Transcribe]`)

## Current limits

- Full transcript is sent to Gemini (no summarization/truncation policy yet).
- Thread history is in-memory only; multi-turn history is not rebuilt after restart.
- No persistent storage for runtime state.
