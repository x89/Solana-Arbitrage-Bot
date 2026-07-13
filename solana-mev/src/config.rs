use std::{collections::HashSet, env, fs, path::Path, str::FromStr};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use solana_pubkey::Pubkey;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub jupiter: JupiterConfig,
    pub scanner: ScannerConfig,
    pub routes: Vec<RouteConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JupiterConfig {
    #[serde(default = "default_jupiter_base_url")]
    pub base_url: String,
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_taker_env")]
    pub taker_env: String,
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScannerConfig {
    #[serde(default = "default_interval_ms")]
    pub interval_ms: u64,
    #[serde(default = "default_min_profit_bps")]
    pub min_profit_bps: i64,
    #[serde(default = "default_slippage_bps")]
    pub slippage_bps: u16,
    #[serde(default)]
    pub estimated_cost_in_start_units: u64,
    #[serde(default = "default_max_accounts")]
    pub max_accounts: u8,
    #[serde(default = "default_true")]
    pub fast_mode: bool,
    #[serde(default = "default_true")]
    pub require_different_venues: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RouteConfig {
    pub name: String,
    pub start_mint: String,
    pub intermediate_mint: String,
    pub amount: u64,
    #[serde(default)]
    pub forward_dexes: Vec<String>,
    #[serde(default)]
    pub return_dexes: Vec<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        let config: Self = toml::from_str(&contents)
            .with_context(|| format!("failed to parse config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn api_key(&self) -> Result<String> {
        read_required_env(&self.jupiter.api_key_env)
    }

    pub fn taker(&self) -> Result<String> {
        let taker = read_required_env(&self.jupiter.taker_env)?;
        validate_pubkey("Jupiter taker", &taker)?;
        Ok(taker)
    }

    fn validate(&self) -> Result<()> {
        if !self.jupiter.base_url.starts_with("https://") {
            bail!("jupiter.base_url must use HTTPS");
        }
        if self.jupiter.api_key_env.trim().is_empty() || self.jupiter.taker_env.trim().is_empty() {
            bail!("Jupiter environment-variable names must not be empty");
        }
        if self.jupiter.request_timeout_ms < 100 {
            bail!("jupiter.request_timeout_ms must be at least 100");
        }
        if self.scanner.interval_ms < 100 {
            bail!("scanner.interval_ms must be at least 100");
        }
        if !(0..=10_000).contains(&self.scanner.min_profit_bps) {
            bail!("scanner.min_profit_bps must be between 0 and 10000");
        }
        if self.scanner.slippage_bps > 10_000 {
            bail!("scanner.slippage_bps must be at most 10000");
        }
        if !(1..=64).contains(&self.scanner.max_accounts) {
            bail!("scanner.max_accounts must be between 1 and 64");
        }

        let enabled_routes: Vec<_> = self.routes.iter().filter(|route| route.enabled).collect();
        if enabled_routes.is_empty() {
            bail!("at least one route must be enabled");
        }

        let mut names = HashSet::new();
        for route in enabled_routes {
            if route.name.trim().is_empty() {
                bail!("route names must not be empty");
            }
            if !names.insert(route.name.as_str()) {
                bail!("duplicate route name: {}", route.name);
            }
            validate_pubkey("start mint", &route.start_mint)
                .with_context(|| format!("invalid route {}", route.name))?;
            validate_pubkey("intermediate mint", &route.intermediate_mint)
                .with_context(|| format!("invalid route {}", route.name))?;
            if route.start_mint == route.intermediate_mint {
                bail!("route {} uses the same mint for both legs", route.name);
            }
            if route.amount == 0 {
                bail!("route {} amount must be greater than zero", route.name);
            }
            if self.scanner.estimated_cost_in_start_units >= route.amount {
                bail!(
                    "estimated cost must be smaller than amount for route {}",
                    route.name
                );
            }
        }

        Ok(())
    }
}

fn read_required_env(name: &str) -> Result<String> {
    let value =
        env::var(name).with_context(|| format!("environment variable {name} is required"))?;
    if value.trim().is_empty() {
        bail!("environment variable {name} must not be empty");
    }
    Ok(value)
}

fn validate_pubkey(label: &str, value: &str) -> Result<()> {
    Pubkey::from_str(value)
        .map(|_| ())
        .with_context(|| format!("{label} is not a valid Solana address: {value}"))
}

fn default_jupiter_base_url() -> String {
    "https://api.jup.ag/swap/v2".to_owned()
}

fn default_api_key_env() -> String {
    "JUPITER_API_KEY".to_owned()
}

fn default_taker_env() -> String {
    "SOLANA_TAKER_PUBKEY".to_owned()
}

const fn default_request_timeout_ms() -> u64 {
    5_000
}

const fn default_interval_ms() -> u64 {
    1_000
}

const fn default_min_profit_bps() -> i64 {
    30
}

const fn default_slippage_bps() -> u16 {
    30
}

const fn default_max_accounts() -> u8 {
    64
}

const fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::{Config, JupiterConfig, RouteConfig, ScannerConfig};

    fn valid_config() -> Config {
        Config {
            jupiter: JupiterConfig {
                base_url: "https://api.jup.ag/swap/v2".to_owned(),
                api_key_env: "JUPITER_API_KEY".to_owned(),
                taker_env: "SOLANA_TAKER_PUBKEY".to_owned(),
                request_timeout_ms: 5_000,
            },
            scanner: ScannerConfig {
                interval_ms: 1_000,
                min_profit_bps: 30,
                slippage_bps: 30,
                estimated_cost_in_start_units: 10_000,
                max_accounts: 64,
                fast_mode: true,
                require_different_venues: true,
            },
            routes: vec![RouteConfig {
                name: "USDC-WSOL-USDC".to_owned(),
                start_mint: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_owned(),
                intermediate_mint: "So11111111111111111111111111111111111111112".to_owned(),
                amount: 100_000_000,
                forward_dexes: vec![],
                return_dexes: vec![],
                enabled: true,
            }],
        }
    }

    #[test]
    fn accepts_safe_valid_config() {
        valid_config().validate().unwrap();
    }

    #[test]
    fn rejects_non_https_jupiter_endpoint() {
        let mut config = valid_config();
        config.jupiter.base_url = "http://api.jup.ag/swap/v2".to_owned();
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_invalid_mint() {
        let mut config = valid_config();
        config.routes[0].start_mint = "not-a-solana-address".to_owned();
        assert!(config.validate().is_err());
    }
}
