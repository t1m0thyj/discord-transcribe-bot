# Usage Guide

Complete [configuration](README.md#configuring) before inviting the bot to a server.

## Discord setup

Create the bot with these OAuth2 scopes:

- `bot`
- `applications.commands`

Grant these recommended permissions:

- View Channels
- Send Messages
- Send Messages in Threads
- Read Message History
- Attach Files
- Create Public Threads
- Connect
- Speak
- Manage Channels, required for `/autojoin` channel suffix changes

Enable the **Message Content Intent** in the Discord Developer Portal for thread Q&A message handling.

## First Call

1. Join the target Discord voice channel.
2. Run `/join` in a text channel in the same guild.
3. Speak normally. The bot replies when it starts receiving usable speech.
4. Use `/status` if speech is not appearing or the call is under load.
5. Run `/leave`, or let the channel empty, to finalize and upload the transcript.

## Commands

| Command | Purpose |
| --- | --- |
| `/join` | Start transcription in your current voice channel. |
| `/status` | Show receive, queue, decode, and transcript health counters. |
| `/log` | Show recent committed transcript lines. |
| `/ask` | Ask the configured AI provider about the active call transcript. |
| `/autojoin` | Mark or unmark a voice channel for automatic transcription. |
| `/leave` | Leave voice and finalize the transcript export. |

## Call lifecycle

The bot supports one active transcription session per guild. After `/join`, it posts a waiting status and begins local speech recognition. On first usable speech, it posts a start confirmation.

When a call ends, the bot uploads a Markdown transcript and creates a thread from that message. Questions in the thread use the committed transcript context. After a restart, the bot can reload the thread context from the transcript attachment, subject to its attachment-size limit.

Optional post-call summaries are configured under `[summary]` in `config.toml` and are disabled by default.

## Autojoin

`/autojoin` appends a suffix to a voice channel name. When a non-bot user enters a marked channel, the bot starts a session automatically. The default suffix is ` [Transcribe]`, configurable with `[discord].autojoin_suffix`.

For autojoin sessions, status and transcript messages go to the first text channel in the same category as the voice channel.

## Transcript Data And AI

The bot stores a JSONL journal locally in `transcripts/` while the call is active and uploads the final Markdown transcript to Discord. Local journal retention is controlled by `[transcription].retention_days`.

Speech recognition remains local. `/ask` and optional summaries send transcript context to the configured AI provider: Gemini is a hosted service; Ollama normally runs locally at the configured URL.

## Health And Recovery

`/status` reports queue depth, ASR work in flight, pending transcript commits, decode failures, real-time factor, decode timing, dropped queued chunks, and unmapped SSRC activity. Use it first when transcription is delayed or missing.

If the bot joins but does not receive usable audio, it attempts voice receive recovery and posts a concise retry status. Session finalization and transcript draining are timeout-bounded; on a drain timeout, export continues with the transcript that was already persisted.