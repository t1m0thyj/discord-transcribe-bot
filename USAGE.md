# Usage Guide

Complete [configuration](README.md#configuring) before inviting the bot to a server.

## Setup Commands

| Command | Purpose |
| --- | --- |
| `transcribe-bot init` | Create `.env` and `config.toml` from the bundled templates without overwriting existing files. Use `cargo run -- init` from a source checkout. |
| `transcribe-bot doctor` | Validate configuration and the local ASR model; for Ollama, also check that the service is reachable and the configured model is installed. Use `cargo run -- doctor` from a source checkout. |
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

## AI Provider Setup

Choose one provider for `/ask` and optional call summaries. Speech transcription remains local for both options.

### Gemini

1. Create an API key in [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Add it to `.env`:

	```text
	GEMINI_API_KEY=your-api-key
	```

3. Set the provider in `config.toml`:

	```toml
	[ai]
	provider = "gemini"
	```

	The optional `[ai.gemini].model` setting defaults to `gemini-flash-latest`.

### Ollama

1. [Install Ollama](https://ollama.com/download), then download a local model. For example, [Gemma 3 4B](https://ollama.com/library/gemma3):

	```bash
	ollama pull gemma3:4b
	```

2. Start the Ollama server in a separate terminal, unless the Ollama desktop app is already serving locally:

	```bash
	ollama serve
	```

3. Set the provider and model in `config.toml`:

	```toml
	[ai]
	provider = "ollama"

	[ai.ollama]
	model = "gemma3:4b"
	base_url = "http://127.0.0.1:11434"
	```

`[ai].request_timeout` is an idle timeout for Ollama's streamed responses.

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
