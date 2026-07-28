# Architecture

Short, current design summary for the Discord live transcription bot.

## Stack

- Rust + tokio
- serenity + songbird receive mode
- sherpa-onnx local ASR
- rubato resampling (48 kHz to 16 kHz mono)
- optional nnnoiseless denoise
- Gemini via reqwest for Q&A

## Runtime flow

1. /join starts a guild-scoped call session.
2. Voice events populate guild-scoped speaker maps and per-user audio buffers.
3. Audio pipeline performs downmix, optional denoise, periodic provisional ASR, and final ASR on silence.
4. Rolling ingest bounds memory for long speech: old chunk is finalized, recent context tail is retained.
5. Utterance revisions are queued and merged by a transcript writer with a small reorder window.
6. /ask and /log wait for brief quiescence, flush pending guild buffers, then read current transcript.
7. /leave (or empty channel auto-finalize) performs settle+flush, disconnects, settle+flush again, then exports transcript and creates a Q&A thread.
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

- Utterance: user_id, start_ts, revision_id, is_final, text
- UserAudioBuffer: audio PCM + segmentation/revision/provisional-stability tracking
- CallSession: voice_channel, text_channel, transcript, started_at, started_mono
- ThreadContext: transcript + turn history
- AppState: active calls, thread contexts, ASR engine, guild-scoped buffers/maps, drain counters

## Ordering and consistency

- ASR work runs concurrently (spawn + spawn_blocking).
- Transcript writes are revision-aware and ordered by start timestamp with a watermark.
- Finalization and interactive reads use inflight/pending counters and explicit flush/drain waits.

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
- ASR_MODEL_DIR (or MOONSHINE_MODEL_DIR fallback)
- ASR_MODEL_FAMILY (optional)
- LIVE_TRANSCRIPT_DEBUG
- ENABLE_DENOISER
- PROVISIONAL_CADENCE_MS
- ROLLING_INGEST_MAX_MS
- ROLLING_INGEST_CONTEXT_MS
- AUTOJOIN_SUFFIX (optional; default `[Transcribe]`)

## Current limits

- Full transcript is sent to Gemini (no summarization/truncation policy yet).
- Thread history is in-memory only; multi-turn history is not rebuilt after restart.
- No persistent storage for runtime state.
