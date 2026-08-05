# Developer Guide

This guide covers local development and maintenance. User-facing setup and commands are in the [usage guide](USAGE.md).

## Local Verification

Install the current stable Rust toolchain. On Windows, use the Visual Studio Build Tools C++ workload before building native dependencies.

Run these checks before submitting a change:

```bash
cargo fmt --check
cargo test --locked
```

The unit tests cover configuration, audio segmentation, DSP helpers, decode-queue behavior, model-layout detection, transcript persistence, export formatting, and AI response parsing. They do not require a Discord connection, an ASR model, or AI credentials.

## Debug logging

Set `[debug].log_live_transcript = true` in `config.toml` to log finalized utterances. Enable debug logs when running the bot:

```bash
RUST_LOG=debug cargo run
```

In PowerShell, use:

```powershell
$env:RUST_LOG = "debug"
cargo run
```

The terminal includes `live transcription` lines after successful ASR decode and before journal persistence.

## Configuration And Generated Files

- `.env` holds `DISCORD_TOKEN`, optional `GEMINI_API_KEY`, and optional `APP_CONFIG_PATH`.
- `config.toml` contains model, AI-provider, audio, transcription, summary, and Discord settings. Start from [config.example.toml](config.example.toml).
- `models/`, `transcripts/`, `.env`, and `config.toml` are local runtime artifacts and are ignored by Git.
- The bot writes a per-call JSONL journal to `transcripts/`; it removes old journals according to `[transcription].retention_days` after a call finalizes.

## Transcript format

Transcript exports are chronological Markdown files with YAML front matter:

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

Each utterance includes its speaker and elapsed call time. The append-only JSONL journal in `transcripts/` is the durable source used to produce this export. The journal is an internal recovery format; integrations should consume the exported Markdown attachment.

## Reference Documents

- [Architecture](ARCHITECTURE.md): runtime flow, state ownership, reliability boundaries, and source map.
- [Usage](USAGE.md): Discord setup, commands, autojoin, transcripts, and health checks.
