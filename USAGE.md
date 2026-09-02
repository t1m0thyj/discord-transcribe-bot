# Usage Guide

Complete [configuration](README.md#configuring) before inviting the bot to a server.

## Setup Commands

| Command | Purpose |
| --- | --- |
| `transcribe-bot init` | Create `.env` and `config.toml` from the bundled templates without overwriting existing files. Use `cargo run -- init` from a source checkout. |
| `transcribe-bot doctor` | Validate configuration and the local ASR model, then contact the AI API to confirm the configured model is listed (without generating text). Transient failures are retried; failure also notes that a non-standard server may still support chat completions without model discovery. Use `cargo run -- doctor` from a source checkout. |
| `transcribe-bot download owner/repo` | Find the `hf` CLI, display the model's Hub information, ask before downloading it into `models/repo`, and verify the detected ASR layout. If `hf` is missing, it prints the `pip` installation command. Use `cargo run -- download owner/repo` from a source checkout. |

When the default `config.toml` is missing, starting the bot performs the same initialization and then exits so the generated files can be configured. `init` is useful for preparing a release-binary directory explicitly.

## Suggested Models

Run `transcribe-bot download owner/repo` for a model below. It displays Hub information, asks before downloading, and verifies the downloaded layout. When `hf` is missing, it prints the `pip` installation command. Set `[asr].model_dir` to the downloaded `models/repo` directory, then run `transcribe-bot doctor`.

| Model | Rank | Approx. download | Notes |
| --- | --- | ---: | --- |
| [Moonshine Base EN INT8](https://huggingface.co/csukuangfj/sherpa-onnx-moonshine-base-en-int8) | Light | 274 MiB | Use when system resources are limited. |
| [Whisper Base EN](https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base.en) | Medium | 432 MiB | English-only Whisper baseline. |
| [Nemo Parakeet TDT 0.6B v2 INT8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8) | Medium | 631 MiB | Recommended starting point for most systems. |
| [Qwen3 ASR 0.6B INT8](https://huggingface.co/csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25) | Heavy | 954 MiB | Higher-capacity ASR model. |
| [Whisper Distil Medium EN](https://huggingface.co/csukuangfj/sherpa-onnx-whisper-distil-medium.en) | Heavy | 2.0 GiB | Largest bundled option. |

## Discord setup

Create the bot with these OAuth2 scopes:

- `bot`
- `applications.commands`

### Bot role permissions

Grant the bot role the following permissions in the text and voice channels where it will operate:

| Permission | Why the bot needs it |
| --- | --- |
| Manage Channels | Required only when using `/autojoin`, which adds or removes the channel-name suffix. |
| View Channels | See the configured text channel and join the selected voice channel. |
| Send Messages | Post command responses and the completed-call transcript message. |
| Create Public Threads | Create the public discussion thread for each transcript. |
| Send Messages in Threads | Reply to transcript-thread questions and post automatic summaries. |
| Attach Files | Upload the final Markdown transcript. |
| Read Message History | Reload transcript attachments when the bot is restarted. |
| Connect | Join the voice channel and receive its audio. |

The bot does not need **Speak**: it receives audio but does not send audio. Apply channel overrides if the bot should be limited to particular meeting channels.

### Member permissions for commands

Members with **Use Application Commands** can use `/status`, `/log`, `/ask`, and `/summary`. `/join` and `/leave` additionally require **Move Members**. `/autojoin` requires **Manage Channels**. Server administrators can further allow or deny individual commands by role, member, or channel in Discord's command/integration settings.

For the post-call transcript thread, participants need **View Channels** on its parent channel and **Send Messages in Threads** to ask the bot questions there.

Enable the **Message Content Intent** in the Discord Developer Portal for thread Q&A message handling.

## AI Provider Setup

Configure one OpenAI-compatible endpoint for `/ask` and optional call summaries. Speech transcription remains local.

### OpenAI-compatible APIs

The bot has one streaming Chat Completions client. Use it with OpenAI, [Gemini's OpenAI-compatible API](https://ai.google.dev/gemini-api/docs/openai), OpenRouter, [Ollama's local OpenAI-compatible endpoint](https://docs.ollama.com/openai), or DeepSeek. It sends requests to `{base_url}/chat/completions` and adds a bearer token only when `[ai].api_key_env` is configured.

For local Ollama, [install Ollama](https://ollama.com/download) and download a model. For example:

	```bash
	ollama pull gemma3:4b
	```

Start the Ollama server in a separate terminal, unless the Ollama desktop app is already serving locally:

	```bash
	ollama serve
	```

Configure it in `config.toml`:

	```toml
	[ai]
	model = "gemma3:4b"
	base_url = "http://127.0.0.1:11434/v1"
	```

For a hosted compatible API, put its key in `.env` as `OPENAI_API_KEY`, then set its base URL and model. For example, [DeepSeek's API](https://api-docs.deepseek.com/) supports the OpenAI Chat Completions format:

	```text
	OPENAI_API_KEY=your-deepseek-api-key
	```

	```toml
	[ai]
	model = "deepseek-v4-flash"
	base_url = "https://api.deepseek.com"
	api_key_env = "OPENAI_API_KEY"
	```

If you prefer a provider-specific variable such as `GEMINI_API_KEY` or `OPENROUTER_API_KEY`, select it without putting the secret in `config.toml`:

	```toml
	[ai]
	model = "gemini-3.7-flash"
	base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
	api_key_env = "GEMINI_API_KEY"
	```

	```toml
	[ai]
	model = "openrouter/free"
	base_url = "https://openrouter.ai/api/v1"
	api_key_env = "OPENROUTER_API_KEY"
	```

`[ai].request_timeout` is the maximum time the API may go without sending response data. Set `[ai].api_key_env` for hosted APIs that require authentication; the named environment variable must then be present and non-empty. When `api_key_env` is omitted, the bot sends no authentication header. AI requests are rate limited by default. Configure these under `[ai.rate_limits]`; set any limit to `0` to disable it.

## First Call

1. Join the target Discord voice channel.
2. Run `/join` in a text channel in the same guild.
3. Speak normally. The bot replies when it starts receiving usable speech.
4. Use `/status` if speech is not appearing or the call is under load.
5. Run `/leave`, or let the channel empty, to finalize and upload the transcript.

## Keeping the bot running

### Linux

Manage the bot with a systemd user service. Replace `/home/you/transcribe-bot` with the directory containing the binary, `.env`, and `config.toml`:

```ini
# ~/.config/systemd/user/transcribe-bot.service
[Unit]
Description=Discord Transcribe Bot

[Service]
WorkingDirectory=/home/you/transcribe-bot
ExecStart=/home/you/transcribe-bot/transcribe-bot
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Create the file, then enable and inspect it with:

```bash
mkdir -p ~/.config/systemd/user
systemctl --user daemon-reload
systemctl --user enable --now transcribe-bot
sudo loginctl enable-linger "$USER" # keep it running after logout and reboot
journalctl --user -u transcribe-bot -f
```

### Other Platforms

* _Windows_ - Use [Task Scheduler](https://learn.microsoft.com/en-us/windows/win32/taskschd/task-scheduler-start-page) to run `transcribe-bot.exe` at startup, set its “Start in” directory, and enable restart on failure.
* _macOS_ - Use a [launchd LaunchAgent](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html) to run the binary at login, set its working directory, and use `KeepAlive` to restart it after failure.

## Commands

| Command | Purpose |
| --- | --- |
| `/join` | Start transcription in your current voice channel. |
| `/status` | Show receive, queue, decode, and transcript health counters. |
| `/log` | Show recent committed transcript lines. |
| `/ask` | Ask the configured AI provider about the active call transcript. |
| `/summary` | Summarize the active call, or generate/retry a summary from a completed transcript thread. |
| `/autojoin` | Mark or unmark a voice channel for automatic transcription. |
| `/leave` | Leave voice and finalize the transcript export. |

## Call lifecycle

The bot supports one active transcription session per guild. After `/join`, it posts a waiting status and begins local speech recognition. On first usable speech, it posts a start confirmation.

When a call ends, the bot uploads a Markdown transcript and creates a thread from that message. Questions in the thread use the committed transcript context; run `/summary` in that thread to generate or retry its meeting summary. After a restart, the bot can reload the thread context from the transcript attachment, subject to its attachment-size limit.

Optional post-call summaries are configured under `[summary]` in `config.toml` and are disabled by default.

## Autojoin

`/autojoin` appends a suffix to a voice channel name. When a non-bot user enters a marked channel, the bot starts a session automatically. The default suffix is ` [Transcribe]`, configurable with `[discord].autojoin_suffix`.

For autojoin sessions, status and transcript messages go to the first text channel in the same category as the voice channel.

For a single-guild installation, optionally set `[discord].autojoin_text_channel_id` to a Discord text-channel ID. It overrides autojoin's channel selection for call status, live transcript, and final transcript exports. `/join` always uses the channel where it was invoked. Leave the setting unset to preserve the normal autojoin behavior.

## Transcript Data And AI

The bot stores a JSONL journal locally in `transcripts/` while the call is active and uploads the final Markdown transcript to Discord. Local journal retention is controlled by `[transcription].retention_days`.

Speech recognition remains local. `/ask` and optional summaries send transcript context to the configured OpenAI-compatible API, which can be local (for example, Ollama) or hosted.

## Health And Recovery

`/status` reports queue depth, ASR work in flight, pending transcript commits, decode failures, real-time factor, decode timing, dropped queued chunks, and unmapped SSRC activity. Use it first when transcription is delayed or missing.

If the bot joins but does not receive usable audio, it attempts voice receive recovery and posts a concise retry status. Session finalization and transcript draining are timeout-bounded; on a drain timeout, export continues with the transcript that was already persisted.
