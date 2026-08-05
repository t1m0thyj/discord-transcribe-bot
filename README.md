# transcribe-bot

Transcribe Discord voice channels locally and answer questions about the finished call.

## What It Does

The bot receives Discord voice audio, transcribes finalized speech locally, and posts a Markdown transcript when the call ends. Its transcript thread supports optional Q&A and summaries through Gemini or Ollama.

## Configuring

1. Create a Discord application and bot. The required scopes, permissions, and Developer Portal settings are in the [usage guide](USAGE.md#discord-setup).
2. Create `.env` from `.env.example`, then set `DISCORD_TOKEN`. Add `GEMINI_API_KEY` only when `[ai].provider = "gemini"`.
3. Create `config.toml` from `config.example.toml`. Set the AI provider and `[asr].model_dir`.
4. Install the Hugging Face CLI and download the default ASR model:

   ```bash
   pip install --upgrade "huggingface_hub[cli]"
   hf download sherpa-onnx/sherpa-onnx-moonshine-base-en-int8 --local-dir models/sherpa-onnx-moonshine-base-en-int8
   ```

   **Note:** For gated models, authenticate first with `hf auth login`.

## Building From Source

Install a current Rust toolchain, then build and run:

```bash
cargo build --release
cargo run --release
```

On Windows, use the Visual Studio Build Tools C++ workload. The workspace includes VS Code build and test tasks for that environment.

## Installing A Prebuilt Binary

Download the binary matching your platform from a published release or the corresponding GitHub Actions build artifact. Put it beside `.env` and `config.toml`, keep the configured model directory available, then run the binary from that directory.

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
