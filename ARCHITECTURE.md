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
2. Voice events populate guild-scoped speaker maps and per-user DSP state.
3. Audio pipeline performs stereo downmix, DC-block high-pass, optional RNNoise denoise, and anti-aliased 48 kHz to 16 kHz resampling (rubato FFT). The resulting 16 kHz stream is fed to both live streaming ASR state and the offline-final capture buffer.
4. A per-session streaming decoder loop owns all `OnlineStream` state. Each 20 ms tick, absent active speakers are fed 320 zero samples so streaming endpoint rules can fire even though Discord stops sending packets on silence.
5. The streaming recognizer emits low-latency `Partial` updates into an in-memory live snapshot map and emits `StreamFinal` utterances on endpoint. Each final shares a `revision_id` with its later offline replacement.
6. Endpointed 16 kHz audio is queued to a bounded offline-final worker. The worker runs one offline decode at a time through the global ASR semaphore and upgrades `StreamFinal` to `OfflineFinal` only if the refinement guard accepts it.
7. Utterance revisions are merged by stage precedence (`Partial < StreamFinal < OfflineFinal`) inside a transcript writer with a small reorder window. Only `StreamFinal`/`OfflineFinal` are persisted to per-session JSONL for crash-safe recovery.
8. `/log` and `/ask` read the published live partial map directly and merge it with committed transcript state for instant snapshots.
9. `/leave` (or empty channel auto-finalize) sends an explicit decoder flush command, waits for the ack and offline queue drain, writes transcript text to local disk, then uploads it and creates a Q&A thread.
10. Thread messages use transcript + in-memory thread history; transcript can be lazily restored from the starter attachment after restart.

## Commands

- /join
- /leave
- /status
- /log
- /ask
- /autojoin

Suffix-based auto-join is implemented: channels ending with the configured marker (default ` [Transcribe]`) are auto-joined when a non-bot user enters.

## Key state

- Utterance: user_id, start_ts, start_offset_ms, revision_id, stage, is_final, text, tokens, token_timestamps_s
- CallSession: voice_channel, text_channel, transcript, started_at, started_mono
- ThreadContext: transcript + turn history
- AppState: active calls, thread contexts, streaming/offline ASR engines, guild-scoped DSP maps, live partial map, queue/health counters

## Ordering and consistency

- Streaming decode is serialized inside the per-session decoder task; offline finalization runs through a bounded queue and a one-permit ASR semaphore.
- Transcript writes are revision-aware and ordered by start timestamp with a watermark.
- Finalization uses an explicit decoder flush ack plus inflight/pending drain waits.

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
- ASR_STREAMING_MODEL_DIR
- ASR_OFFLINE_MODEL_DIR
- ASR_MODEL_FAMILY (optional)
- LIVE_TRANSCRIPT_DEBUG
- ENABLE_DENOISER
- AUTOJOIN_SUFFIX (optional; default `[Transcribe]`)

## Current limits

- Full transcript is sent to Gemini (no summarization/truncation policy yet).
- Thread history is in-memory only; multi-turn history is not rebuilt after restart.
- No persistent storage for runtime state.
