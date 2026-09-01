# transcribe-bot

Transcribe Discord voice channels locally and use AI to summarize the conversation or answer follow-up questions.

## Install

### Prebuilt binary

From the directory where the bot should run, download the matching artifact from the latest successful GitHub Actions build:

```bash
curl -fsSL https://raw.githubusercontent.com/t1m0thyj/discord-transcribe-bot/main/install.sh | sh
```

The installer extracts the binary into the current directory and removes the downloaded archive. Run `./transcribe-bot init` there, complete [configuration](#configure-and-run), then start the binary from that directory. On Windows, use `./transcribe-bot.exe`.

### Build from source

Install a current Rust toolchain, then build:

```bash
cargo build --release
```

On Windows, use the Visual Studio Build Tools C++ workload. The workspace includes VS Code build and test tasks for that environment.

## Configure and run

1. Run `./transcribe-bot init` in the directory where the bot will run, or `cargo run -- init` from a source checkout. It creates `.env` and `config.toml` without overwriting existing files.
2. Create a Discord application and bot. The required scopes, permissions, and Developer Portal settings are in the [usage guide](USAGE.md#discord-setup).
3. Choose an [AI provider](USAGE.md#ai-provider-setup) and a [suggested ASR model](USAGE.md#suggested-models). Start with Parakeet unless the system is low-resource, in which case choose Moonshine.
4. Set `DISCORD_TOKEN` in `.env`, add the selected provider's API key when needed, and configure the chosen provider in `config.toml`.
5. Download the selected ASR model. For the recommended starting model:

   ```bash
   ./transcribe-bot download csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8
   ```

   From a source checkout, use `cargo run -- download csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` instead. After the model has been downloaded, set `[asr].model_dir` in `config.toml` to that downloaded `models/repo` directory.

   **Note:** For gated models, authenticate first with `hf auth login`.

6. Run `./transcribe-bot doctor` (or `cargo run -- doctor` from source) to validate the configuration, ASR model, and configured AI provider.
7. Start the bot with `./transcribe-bot`, or `cargo run --release` from a source checkout. On Windows, use `./transcribe-bot.exe`.

## How It Works

Use `/join` in a voice channel to start a call. The bot segments and transcribes each participant locally, persists finalized lines during the call, then uploads a Markdown transcript at the end. It creates a thread from that transcript, where participants can ask questions about the call.

The optional `/autojoin` command marks a voice channel to begin transcription when someone joins it. One call session runs per guild at a time.

## Technology

- Rust, Tokio, Serenity, and Songbird for the Discord bot and voice receive path.
- sherpa-onnx for local speech recognition.
- An OpenAI-compatible API for optional transcript Q&A and summaries, such as Gemini, OpenRouter, or Ollama.
- Append-only JSONL journals and Markdown exports for durable call transcripts.

## Further Reading

- [Usage guide](USAGE.md): Discord OAuth setup, permissions, commands, autojoin, recovery, and call lifecycle.
- [Developer guide](DEVELOPERS.md): debug logging and transcript format.
- [Architecture](ARCHITECTURE.md): audio pipeline, persistence, runtime state, and source map.
