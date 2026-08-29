mod ai;
mod app;
mod asr;
mod cli;
mod config;

use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use serenity::all::{GatewayIntents, Interaction};
use serenity::async_trait;
use serenity::prelude::*;
use songbird::SerenityInit;

use app::AppState;
use config::AppConfig;

struct Handler {
    state: Arc<AppState>,
}

#[async_trait]
impl EventHandler for Handler {
    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        match interaction {
            Interaction::Command(command) => {
                app::handle_slash_command(&ctx, &self.state, command).await;
            }
            Interaction::Autocomplete(command) => {
                app::handle_autocomplete(&ctx, &self.state, command).await;
            }
            _ => {}
        }
    }

    async fn message(&self, ctx: Context, msg: serenity::all::Message) {
        app::handle_message(&ctx, &self.state, msg).await;
    }

    async fn ready(&self, ctx: Context, ready: serenity::all::Ready) {
        tracing::info!("Connected as {}", ready.user.name);
        if let Err(e) = app::register_commands(&ctx).await {
            tracing::error!("failed to register slash commands: {e:#}");
        }
    }

    async fn voice_state_update(
        &self,
        ctx: Context,
        old: Option<serenity::all::VoiceState>,
        new: serenity::all::VoiceState,
    ) {
        if let Some(guild_id) = new.guild_id {
            app::maybe_autojoin_on_voice_state(
                &ctx,
                &self.state,
                guild_id,
                old.clone(),
                new.clone(),
            )
            .await;

            if let Err(e) =
                app::maybe_finalize_on_empty_voice_channel(&ctx, &self.state, guild_id, old, new)
                    .await
            {
                tracing::warn!("voice-state finalize check failed: {e:#}");
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    init_rustls_crypto_provider();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build tokio runtime")?;

    runtime.block_on(run())
}

fn init_rustls_crypto_provider() {
    // Required when reqwest uses rustls-no-provider.
    // Ignore "already installed" in case dependencies initialize first.
    let _ = rustls::crypto::ring::default_provider().install_default();
}

async fn run() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new(
                    "warn,transcribe_bot=info,serenity=warn,songbird=warn,songbird::driver::tasks::udp_rx=warn,songbird::driver::tasks::udp_rx::ssrc_state=warn",
                )
            }),
        )
        .init();

    match cli::command_from_env()? {
        cli::Command::Init => {
            cli::initialize_current_directory()?;
            return Ok(());
        }
        cli::Command::Doctor => {
            cli::run_doctor().await?;
            return Ok(());
        }
        cli::Command::Download { repo_id } => {
            cli::download_model(&repo_id)?;
            return Ok(());
        }
        cli::Command::Help => return Ok(()),
        cli::Command::Run => {}
    }

    if should_initialize_automatically() {
        cli::initialize_current_directory()?;
        anyhow::bail!(
            "configuration was created; configure .env and config.toml, then run transcribe-bot again"
        );
    }

    let cfg = AppConfig::from_env()?;

    let state = Arc::new(AppState::new(cfg.clone()).context("initializing app state")?);

    let intents = GatewayIntents::GUILDS
        | GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT
        | GatewayIntents::GUILD_VOICE_STATES;

    let songbird = songbird::Songbird::serenity();
    songbird.set_config(songbird::Config::default().decode_mode(
        songbird::driver::DecodeMode::Decode(songbird::driver::DecodeConfig::default()),
    ));

    let mut client = serenity::Client::builder(&cfg.discord_token, intents)
        .event_handler(Handler { state })
        .register_songbird_with(songbird)
        .await?;

    client.start().await?;
    Ok(())
}

fn should_initialize_automatically() -> bool {
    let config_path = &config::resolve_config_path();
    config_path == Path::new("config.toml") && !config_path.exists()
}
