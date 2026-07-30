# transcribe-bot

Discord voice transcription + live Q&A bot in Rust.

## Quick start

1. Copy `.env.example` to `.env` and fill values.
   - Use a currently available Gemini model, e.g. `gemini-3.6-flash`.
   - Alternative moving target alias: `gemini-flash-latest`.
2. Ensure both ASR model directories exist.
	- Streaming partials/endpointer: `ASR_STREAMING_MODEL_DIR`
	- Authoritative offline finalizer: `ASR_OFFLINE_MODEL_DIR`
	- Example offline paths in this repo: `models/sherpa-onnx-moonshine-base-en-int8`, `models/sherpa-onnx-whisper-base.en`
3. Run:

```bash
cargo run
```

## Debug logging

To print live transcription lines to the console:

1. Set `LIVE_TRANSCRIPT_DEBUG=true` in `.env`.
2. Run with debug logs enabled, for example:

```bash
RUST_LOG=debug cargo run
```

You will see `live transcription` log lines in the terminal as utterances are transcribed.

## Transcript format

Transcript exports are chronological and Zoom-like:

```text
Meeting transcript
Started: 2026-07-25 20:15:04 UTC
Format: [HH:MM:SS] Speaker: text

[00:00:02] Alice: Let us begin.
[00:00:05] Bob: Sounds good.
```

`HH:MM:SS` is elapsed call time from the first captured utterance.

When an utterance could not be replaced by offline refinement, it is labeled as `[stream-final]`.

## After the call ends

- The bot posts the full transcript as a `.txt` file attachment.
- The bot creates a thread from that transcript message.
- Questions posted in that thread are answered using the transcript context.
- If the bot restarts and thread context is not in memory, it lazily reloads the transcript from the starter message attachment on the first follow-up message.

## Commands

- `/join` - bot joins your current voice channel in that guild.
- `/status` - shows current receive/transcription health and queue state.
- `/log` - prints a recent transcript snapshot.
- `/ask` - asks Gemini about the current live transcript.
- `/autojoin` - marks/unmarks your current voice channel (or an explicitly mentioned voice channel) by suffix so the bot auto-starts there later (default marker: `[Transcribe]`).
- `/leave` - bot leaves and finalizes call transcript export.

## Runtime behavior

- The bot supports one active transcription session per guild at a time.
- The bot uses a hybrid ASR path: a streaming recognizer produces low-latency live text and endpointing, and an offline recognizer overwrites each endpointed segment with the authoritative final transcript when accepted by the refinement guard.
- After `/join`, the bot posts a short "Listening and waiting for speech..." message.
- `/autojoin` renames the voice channel with a configurable suffix (default: ` [Transcribe]`); when a non-bot user joins that marked channel, the bot auto-starts transcription.
- For autojoin sessions, status/transcript messages go to the first text channel in the same category as the voice channel.
- Set `AUTOJOIN_SUFFIX` in `.env` to customize the marker text (for example: `[Transcribe]` or `🎙️`).
- Once first usable speech is decoded, it posts "Started transcribing in <#voice-channel>." (once per call).
- If receive starts unhealthy (no decoded audio after join), the bot attempts automatic voice receive recovery and posts concise retry status.
- `/log` and `/ask` read the published live partial map directly; they do not clone PCM or run extra ASR work.
- Partials are kept in memory for live snapshots (`/log`, `/ask`) and are not persisted to JSONL.

## Config summary

- `ASR_STREAMING_MODEL_DIR` points at the sherpa-onnx streaming transducer used for live partials and endpointing.
- `ASR_OFFLINE_MODEL_DIR` points at the stronger offline recognizer used for authoritative finals.
- `ENABLE_DENOISER` defaults to `false`; enable it only if you explicitly want RNNoise in the hybrid path.

## Accuracy Caveat (Discord)

Discord clients can apply local VAD/noise suppression before audio is transmitted. If leading phonemes are clipped on the sender side, server-side buffering cannot recover them. Practical mitigation is social: ask speakers to lower Discord input sensitivity or use push-to-talk with a generous hold window.

## Discord OAuth2 setup

Required OAuth2 scopes:

- `bot`
- `applications.commands`

Recommended bot permissions:

- View Channels
- Send Messages
- Send Messages in Threads
- Read Message History
- Attach Files
- Create Public Threads
- Connect
- Speak
- Manage Channels (required for `/autojoin` channel suffix rename)

Developer Portal setting:

- Enable Message Content Intent (required for thread Q&A message handling)
