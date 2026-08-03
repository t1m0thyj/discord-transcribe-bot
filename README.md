# transcribe-bot

Discord voice transcription + live Q&A bot in Rust.

## Quick start

1. Copy `.env.example` to `.env` and fill values.
   - Set `AI_PROVIDER=ollama|gemini` (template default: `ollama`).
   - Ollama: run `ollama serve`, set `OLLAMA_MODEL`, optionally change `OLLAMA_BASE_URL`.
   - Gemini: set `GEMINI_API_KEY`; `GEMINI_MODEL` is optional (default: `gemini-flash-latest`).
2. Ensure your ASR model directory exists under `ASR_MODEL_DIR`.
	- Example paths in this repo: `models/sherpa-onnx-moonshine-base-en-int8`, `models/sherpa-onnx-whisper-base.en`
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
---
title: "Transcript 2026-07-25 20:15:04 UTC"
type: meeting
date: 2026-07-25T20:15:04Z
duration: "48m 12s"
source: discord
status: complete
attendees:
   - Alice
   - Bob
---

## Transcript

[Alice 0:03] Are we ready to deploy?
[Bob 0:08] I will run the final tests tonight.
```

Each utterance is prefixed with speaker name and elapsed call time.

## After the call ends

- The bot posts the full transcript as a Markdown (`.md`) file attachment.
- The bot creates a thread from that transcript message.
- Questions posted in that thread are answered using the transcript context.
- If the bot restarts and thread context is not in memory, it lazily reloads the transcript from the starter message attachment on the first follow-up message.
- Optional auto-summary is available via env vars (off by default for privacy).
- `POST_CALL_SUMMARY_ENABLED=true`
- `POST_CALL_SUMMARY_POST_IN_THREAD=true` (default)
- `POST_CALL_SUMMARY_INCLUDE_IN_MARKDOWN=false` (default)

## Commands

- `/join` - bot joins your current voice channel in that guild.
- `/status` - shows current receive/transcription health and queue state.
- `/log` - prints recent committed transcript lines.
- `/ask` - asks the configured AI provider about the current committed transcript.
- `/autojoin` - marks/unmarks your current voice channel (or an explicitly mentioned voice channel) by suffix so the bot auto-starts there later (default marker: `[Transcribe]`).
- `/leave` - bot leaves and finalizes call transcript export.

## Runtime behavior

- The bot supports one active transcription session per guild at a time.
- After `/join`, the bot posts a short "Listening and waiting for speech..." message.
- `/autojoin` renames the voice channel with a configurable suffix (default: ` [Transcribe]`); when a non-bot user joins that marked channel, the bot auto-starts transcription.
- For autojoin sessions, status/transcript messages go to the first text channel in the same category as the voice channel.
- Set `AUTOJOIN_SUFFIX` in `.env` to customize the marker text (for example: `[Transcribe]` or `🎙️`).
- Once first usable speech is decoded, it posts "Started transcribing in <#voice-channel>." (once per call).
- If receive starts unhealthy (no decoded audio after join), the bot attempts automatic voice receive recovery and posts concise retry status.

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
