# transcribe-bot

Transcribe Discord voice channels locally and use AI to summarize the conversation or answer follow-up questions.

## Configuring

1. Run `transcribe-bot init` in the directory where the bot will run, or `cargo run -- init` from a source checkout. It creates `.env` and `config.toml` without overwriting existing files.
2. Create a Discord application and bot. The required scopes, permissions, and Developer Portal settings are in the [usage guide](USAGE.md#discord-setup).
3. Set `DISCORD_TOKEN` in `.env`. Add `GEMINI_API_KEY` when using Gemini; see [AI provider setup](USAGE.md#ai-provider-setup).
4. Select an AI provider and set `[asr].model_dir` in `config.toml`.
5. Install the Hugging Face CLI and download the default ASR model:

   ```bash
   pip install --upgrade "huggingface_hub[cli]"
   hf download sherpa-onnx/sherpa-onnx-moonshine-base-en-int8 --local-dir models/sherpa-onnx-moonshine-base-en-int8
   ```

   **Note:** For gated models, authenticate first with `hf auth login`.

6. Run `transcribe-bot doctor` (or `cargo run -- doctor` from source) to validate the configuration, ASR model, and configured AI provider before starting the bot.

## Building From Source

Install a current Rust toolchain, then build and run:

```bash
cargo build --release
cargo run --release
```

On Windows, use the Visual Studio Build Tools C++ workload. The workspace includes VS Code build and test tasks for that environment.

## Installing A Prebuilt Binary

Download the binary matching your platform from a published release or the corresponding GitHub Actions build artifact and put it in an empty directory. Run `transcribe-bot init` there, complete [configuration](#configuring), then start the binary from that directory.

## How It Works

Use `/join` in a voice channel to start a call. The bot segments and transcribes each participant locally, persists finalized lines during the call, then uploads a Markdown transcript at the end. It creates a thread from that transcript, where participants can ask questions about the call.

The optional `/autojoin` command marks a voice channel to begin transcription when someone joins it. One call session runs per guild at a time.

## Technology

- Rust, Tokio, Serenity, and Songbird for the Discord bot and voice receive path.
- sherpa-onnx for local speech recognition.
- Gemini or Ollama for optional transcript Q&A and summaries.
- Append-only JSONL journals and Markdown exports for durable call transcripts.

## Further Reading

- [Usage guide](USAGE.md): Discord OAuth setup, permissions, commands, autojoin, recovery, and call lifecycle.
- [Developer guide](DEVELOPERS.md): debug logging and transcript format.
- [Architecture](ARCHITECTURE.md): audio pipeline, persistence, runtime state, and source map.
