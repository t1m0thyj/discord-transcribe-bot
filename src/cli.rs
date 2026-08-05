use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context as _;

use crate::ai::AiProviderConfig;
use crate::asr::{resolve_model_dir, validate_model_layout};
use crate::config::{self, AppConfig};

const ENV_TEMPLATE: &str = include_str!("../.env.example");
const CONFIG_TEMPLATE: &str = include_str!("../config.example.toml");

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Command {
    Run,
    Init,
    Doctor,
    Help,
}

pub(crate) fn command_from_env() -> anyhow::Result<Command> {
    parse_command(std::env::args().skip(1))
}

fn parse_command(arguments: impl IntoIterator<Item = String>) -> anyhow::Result<Command> {
    let arguments: Vec<_> = arguments.into_iter().collect();
    match arguments.as_slice() {
        [] => Ok(Command::Run),
        [command] if command == "init" => Ok(Command::Init),
        [command] if command == "doctor" => Ok(Command::Doctor),
        [command] if command == "--help" || command == "-h" || command == "help" => {
            println!("Usage: transcribe-bot [init|doctor]");
            Ok(Command::Help)
        }
        _ => anyhow::bail!("usage: transcribe-bot [init|doctor]"),
    }
}

pub(crate) fn should_initialize_automatically() -> bool {
    let config_path = PathBuf::from(config::resolve_config_path());
    config_path == Path::new("config.toml") && !config_path.exists()
}

pub(crate) fn initialize_current_directory() -> anyhow::Result<()> {
    let config_path = PathBuf::from(config::resolve_config_path());
    let env_path = PathBuf::from(".env");
    let config_created = write_template_if_missing(&config_path, CONFIG_TEMPLATE)?;
    let env_created = write_template_if_missing(&env_path, ENV_TEMPLATE)?;

    if config_created {
        println!("Created {}.", config_path.display());
    } else {
        println!("Kept existing {}.", config_path.display());
    }
    if env_created {
        println!("Created {}.", env_path.display());
    } else {
        println!("Kept existing {}.", env_path.display());
    }

    println!("\nNext steps:");
    println!("1. Set DISCORD_TOKEN in .env.");
    println!(
        "2. Configure Gemini or Ollama in {}.",
        config_path.display()
    );
    println!("3. Set [asr].model_dir and download that ASR model.");
    println!("4. Run transcribe-bot doctor, then transcribe-bot.");
    Ok(())
}

fn write_template_if_missing(path: &Path, template: &str) -> anyhow::Result<bool> {
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            file.write_all(template.as_bytes())
                .with_context(|| format!("failed to write {}", path.display()))?;
            Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(error).with_context(|| format!("failed to create {}", path.display())),
    }
}

pub(crate) async fn run_doctor() -> anyhow::Result<()> {
    let mut failures = Vec::new();
    let config_path = PathBuf::from(config::resolve_config_path());
    check(
        config_path.is_file(),
        format!("configuration: {}", config_path.display()),
        &mut failures,
    );

    let cfg = match AppConfig::from_env() {
        Ok(cfg) => {
            println!("ok: configuration values are valid");
            Some(cfg)
        }
        Err(error) => {
            println!("error: configuration values are invalid: {error:#}");
            failures.push("configuration values".to_string());
            None
        }
    };

    if let Some(cfg) = cfg {
        check_model(&cfg, &mut failures);
        check_ai_provider(&cfg, &mut failures).await;
    }

    if failures.is_empty() {
        println!("\nDoctor found no problems.");
        Ok(())
    } else {
        anyhow::bail!("doctor found problems: {}", failures.join(", "))
    }
}

fn check_model(cfg: &AppConfig, failures: &mut Vec<String>) {
    let path = match resolve_model_dir(&cfg.asr.model_dir) {
        Ok(path) => path,
        Err(error) => {
            println!("error: ASR model path: {error:#}");
            failures.push("ASR model path".to_string());
            return;
        }
    };
    if !path.is_dir() {
        println!(
            "error: ASR model directory does not exist: {}",
            path.display()
        );
        failures.push("ASR model directory".to_string());
        return;
    }

    match validate_model_layout(&path, cfg.asr.model_family.as_deref()) {
        Ok(family) => println!("ok: ASR model: {family} ({})", path.display()),
        Err(error) => {
            println!("error: ASR model layout: {error:#}");
            failures.push("ASR model layout".to_string());
        }
    }
}

async fn check_ai_provider(cfg: &AppConfig, failures: &mut Vec<String>) {
    match &cfg.ai.provider {
        AiProviderConfig::Gemini { model, .. } => {
            println!("ok: Gemini configured with model {model}");
        }
        AiProviderConfig::Ollama { base_url, model } => {
            let endpoint = format!("{}/api/tags", base_url.trim_end_matches('/'));
            let client = match reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(cfg.ai.request_timeout))
                .build()
            {
                Ok(client) => client,
                Err(error) => {
                    println!("error: failed to create Ollama client: {error:#}");
                    failures.push("Ollama client".to_string());
                    return;
                }
            };

            match client.get(&endpoint).send().await {
                Ok(response) if response.status().is_success() => {
                    match response.json::<serde_json::Value>().await {
                        Ok(payload) if ollama_model_is_available(&payload, model) => {
                            println!("ok: Ollama is reachable at {base_url} and has model {model}");
                        }
                        Ok(_) => {
                            println!("error: Ollama is reachable at {base_url}, but model {model} is not installed");
                            failures.push("Ollama model".to_string());
                        }
                        Err(error) => {
                            println!(
                                "error: failed to read Ollama model list from {endpoint}: {error}"
                            );
                            failures.push("Ollama service".to_string());
                        }
                    }
                }
                Ok(response) => {
                    println!(
                        "error: Ollama returned {} from {endpoint}",
                        response.status()
                    );
                    failures.push("Ollama service".to_string());
                }
                Err(error) => {
                    println!("error: Ollama is not reachable at {base_url}: {error}");
                    failures.push("Ollama service".to_string());
                }
            }
        }
    }
}

fn ollama_model_is_available(payload: &serde_json::Value, configured_model: &str) -> bool {
    payload
        .get("models")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.get("name").and_then(serde_json::Value::as_str))
        .any(|name| name == configured_model)
}

fn check(condition: bool, label: String, failures: &mut Vec<String>) {
    if condition {
        println!("ok: {label}");
    } else {
        println!("error: missing {label}");
        failures.push(label);
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{ollama_model_is_available, parse_command, write_template_if_missing, Command};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let unique = format!(
                "transcribe-bot-cli-{name}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("valid clock")
                    .as_nanos()
            );
            let path = std::env::temp_dir().join(unique);
            fs::create_dir_all(&path).expect("create temp directory");
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn parser_recognizes_operational_commands() {
        assert_eq!(parse_command(Vec::<String>::new()).unwrap(), Command::Run);
        assert_eq!(
            parse_command(vec!["init".to_string()]).unwrap(),
            Command::Init
        );
        assert_eq!(
            parse_command(vec!["doctor".to_string()]).unwrap(),
            Command::Doctor
        );
        assert_eq!(
            parse_command(vec!["--help".to_string()]).unwrap(),
            Command::Help
        );
        assert!(parse_command(vec!["unknown".to_string()]).is_err());
    }

    #[test]
    fn template_writer_creates_once_without_overwriting() {
        let directory = TempDir::new("template");
        let path = directory.0.join("config.toml");

        assert!(write_template_if_missing(&path, "first").unwrap());
        assert_eq!(fs::read_to_string(&path).unwrap(), "first");
        assert!(!write_template_if_missing(&path, "second").unwrap());
        assert_eq!(fs::read_to_string(&path).unwrap(), "first");
    }

    #[test]
    fn ollama_model_check_requires_the_configured_model() {
        let payload = serde_json::json!({
            "models": [{"name": "gemma3:4b"}, {"name": "llama3.1:8b"}]
        });
        assert!(ollama_model_is_available(&payload, "gemma3:4b"));
        assert!(!ollama_model_is_available(&payload, "missing:latest"));
    }
}
