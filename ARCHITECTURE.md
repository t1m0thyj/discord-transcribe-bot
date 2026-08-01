# Architecture

Practical architecture overview for the Discord live transcription bot, with enough detail to maintain and debug it.

## At a glance

- One active call session per guild.
- Speech is transcribed locally with a single ASR model.
- Only finalized utterances are committed (no revision stream).
- Commits are journaled incrementally to disk.
- End-of-call export posts transcript and opens a Q&A thread.

## Stack

- Rust + tokio runtime.
- serenity + songbird for Discord events and voice receive.
- sherpa-onnx for local ASR.
- Optional nnnoiseless denoiser.
- Gemini (via reqwest) for transcript Q&A.

## Runtime flow

1. /join creates the guild call session and attaches voice handlers.
2. Voice events map SSRC to users and feed per-user stream state.
3. Audio is cleaned, VAD-gated, segmented, and dispatched for ASR decode.
4. Finalized utterances are committed in receive order.
5. Each commit is appended to a per-session JSONL journal in transcripts/.
6. /log and /ask read committed transcript state (non-destructive).
7. /leave (or empty channel finalize) settles and flushes with bounded waits, writes a local transcript file, uploads to Discord, and creates a thread.

## Audio and transcription path

- Per-user ingest state combines denoiser + rolling buffer.
- Pipeline stages: downmix, high-pass, optional denoise, 48k->16k resample, VAD, endpointing.
- Rolling ingest bounds long speech segments and dispatches chunks through a bounded decode queue.
- Decode rejection and dispatch gating drop implausible/low-value chunks early.

## Ordering and consistency

- Transcript commits are append-only and persisted incrementally to JSONL.
- Finalized utterances are written in receive order.
- Session shutdown uses bounded settle and drain waits to avoid hanging forever.
- Export reads persisted transcript data so finalized lines survive transient Discord upload failures.

## Commands

- /join: start transcription in your current voice channel.
- /status: show receive/transcription health counters.
- /log: print recent committed transcript lines.
- /ask: ask Gemini about the committed transcript.
- /autojoin: mark/unmark a channel by suffix for auto-start.
- /leave: finalize and export transcript.

Autojoin is suffix-based: channels ending with the configured marker (default [Transcribe]) are auto-joined when a non-bot user enters.

## Core state

- CallSession: transcript, channel IDs, session timestamps.
- GuildRuntime: per-guild sender, health counters, recovery lock.
- UserStreamState: per-user DSP state and rolling audio buffer.
- ThreadContext: transcript snapshot and bounded Q&A turn history.

## Reliability behavior

- Startup watchdog recovers if join succeeds but usable audio does not.
- Steady-state watchdog recovers stalled receive paths.
- Finalization waits are timeout-bounded to avoid indefinite hangs.
- Incremental JSONL journaling protects transcript data across crashes or failed uploads.

## Main modules

- src/main.rs: startup, intents, event routing.
- src/config.rs: env/config parsing.
- src/audio.rs: voice ingest, DSP path, segmentation, decode dispatch.
- src/transcription.rs: ASR engine setup, DSP helpers, transcript writer.
- src/app/mod.rs: shared app state and command/message routing.
- src/app/commands.rs: slash command handlers and session start.
- src/app/session.rs: finalize/recovery/watchdogs/export/thread context.
- src/gemini.rs: transcript Q&A API calls.

## Important config knobs

- DISCORD_TOKEN
- GEMINI_API_KEY
- GEMINI_MODEL
- ASR_MODEL_DIR
- ASR_MODEL_FAMILY (optional)
- LIVE_TRANSCRIPT_DEBUG
- ENABLE_DENOISER
- ENDPOINT_SILENCE_MS
- ROLLING_INGEST_MAX_MS
- ROLLING_INGEST_CONTEXT_MS
- AUTOJOIN_SUFFIX (optional; default [Transcribe])

## Current limits

- Q&A context is bounded (no full long-term conversation memory).
- Thread turn history is in-memory and intentionally limited.
- Runtime state is not persisted across restarts (transcript journal is persisted).
- Single-pass finalized transcription favors stability over immediate partial text updates.
