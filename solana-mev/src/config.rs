use std::{collections::HashSet, env, fs, path::Path, str::FromStr};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use solana_pubkey::Pubkey;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema_version: u16,
    pub jupiter: JupiterConfig,
    pub scanner: ScannerConfig,
    pub routes: Vec<RouteConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JupiterConfig {
    #[serde(default = "default_jupiter_base_url")]
    pub base_url: String,
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_taker_env")]
    pub taker_env: String,
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,
    #[serde(default = "default_min_request_interval_ms")]
    pub min_request_interval_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScannerConfig {
    #[serde(default = "default_interval_ms")]
    pub interval_ms: u64,
    #[serde(default = "default_min_profit_bps")]
    pub min_profit_bps: i64,
    #[serde(default = "default_slippage_bps")]
    pub slippage_bps: u16,
    #[serde(default = "default_max_accounts")]
    pub max_accounts: u8,
    #[serde(default = "default_true")]
    pub fast_mode: bool,
    #[serde(default = "default_true")]
    pub require_different_venues: bool,
    #[serde(default = "default_max_cycle_duration_ms")]
    pub max_cycle_duration_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteConfig {
    pub name: String,
    pub start_mint: String,
    pub intermediate_mint: String,
    pub amount: u64,
    #[serde(default)]
    pub estimated_cost_in_start_units: u64,
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
        if self.schema_version != 1 {
            bail!(
                "unsupported config schema_version {}; expected 1",
                self.schema_version
            );
        }
        let base_url =
            reqwest::Url::parse(&self.jupiter.base_url).context("invalid jupiter.base_url")?;
        if base_url.scheme() != "https" {
            bail!("jupiter.base_url must use HTTPS");
        }
        if base_url.host_str() != Some("api.jup.ag")
            || base_url.path().trim_end_matches('/') != "/swap/v2"
            || !base_url.username().is_empty()
            || base_url.password().is_some()
            || base_url.query().is_some()
            || base_url.fragment().is_some()
        {
            bail!("jupiter.base_url must be the official https://api.jup.ag/swap/v2 endpoint");
        }
        if !is_trimmed_nonempty(&self.jupiter.api_key_env)
            || !is_trimmed_nonempty(&self.jupiter.taker_env)
        {
            bail!("Jupiter environment-variable names must be nonempty and trimmed");
        }
        if self.jupiter.request_timeout_ms < 100 {
            bail!("jupiter.request_timeout_ms must be at least 100");
        }
        if self.jupiter.request_timeout_ms > 60_000 {
            bail!("jupiter.request_timeout_ms must be at most 60000");
        }
        if self.jupiter.min_request_interval_ms > 60_000 {
            bail!("jupiter.min_request_interval_ms must be at most 60000");
        }
        if self.scanner.interval_ms < 100 {
            bail!("scanner.interval_ms must be at least 100");
        }
        if !(0..=10_000).contains(&self.scanner.min_profit_bps) {
            bail!("scanner.min_profit_bps must be between 0 and 10000");
        }
        if self.scanner.slippage_bps >= 10_000 {
            bail!("scanner.slippage_bps must be less than 10000");
        }
        if !(1..=64).contains(&self.scanner.max_accounts) {
            bail!("scanner.max_accounts must be between 1 and 64");
        }
        if !(100..=300_000).contains(&self.scanner.max_cycle_duration_ms) {
            bail!("scanner.max_cycle_duration_ms must be between 100 and 300000");
        }

        let enabled_routes: Vec<_> = self.routes.iter().filter(|route| route.enabled).collect();
        if enabled_routes.is_empty() {
            bail!("at least one route must be enabled");
        }

        let mut names = HashSet::new();
        for route in enabled_routes {
            if !is_trimmed_nonempty(&route.name) {
                bail!("route names must be nonempty and trimmed");
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
            if route.estimated_cost_in_start_units >= route.amount {
                bail!(
                    "estimated cost must be smaller than amount for route {}",
                    route.name
                );
            }
            validate_dex_labels("forward_dexes", &route.name, &route.forward_dexes)?;
            validate_dex_labels("return_dexes", &route.name, &route.return_dexes)?;
        }

        Ok(())
    }
}

fn is_trimmed_nonempty(value: &str) -> bool {
    !value.is_empty() && value.trim() == value
}

fn validate_dex_labels(field: &str, route: &str, labels: &[String]) -> Result<()> {
    let mut unique = HashSet::new();
    for label in labels {
        if !is_trimmed_nonempty(label) || label.contains(',') {
            bail!("route {route} has an invalid {field} entry: {label:?}");
        }
        if !unique.insert(label) {
            bail!("route {route} has a duplicate {field} entry: {label}");
        }
    }
    Ok(())
}

fn read_required_env(name: &str) -> Result<String> {
    let value =
        env::var(name).with_context(|| format!("environment variable {name} is required"))?;
    if !is_trimmed_nonempty(&value) {
        bail!("environment variable {name} must be nonempty and trimmed");
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

const fn default_min_request_interval_ms() -> u64 {
    1_100
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

const fn default_max_cycle_duration_ms() -> u64 {
    15_000
}

const fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::{Config, JupiterConfig, RouteConfig, ScannerConfig};

    fn valid_config() -> Config {
        Config {
            schema_version: 1,
            jupiter: JupiterConfig {
                base_url: "https://api.jup.ag/swap/v2".to_owned(),
                api_key_env: "JUPITER_API_KEY".to_owned(),
                taker_env: "SOLANA_TAKER_PUBKEY".to_owned(),
                request_timeout_ms: 5_000,
                min_request_interval_ms: 1_100,
            },
            scanner: ScannerConfig {
                interval_ms: 1_000,
                min_profit_bps: 30,
                slippage_bps: 30,
                max_accounts: 64,
                fast_mode: true,
                require_different_venues: true,
                max_cycle_duration_ms: 15_000,
            },
            routes: vec![RouteConfig {
                name: "USDC-WSOL-USDC".to_owned(),
                start_mint: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_owned(),
                intermediate_mint: "So11111111111111111111111111111111111111112".to_owned(),
                amount: 100_000_000,
                estimated_cost_in_start_units: 10_000,
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

        config.jupiter.base_url = "https://example.com/swap/v2".to_owned();
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_invalid_mint() {
        let mut config = valid_config();
        config.routes[0].start_mint = "not-a-solana-address".to_owned();
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_ambiguous_environment_names_and_dex_labels() {
        let mut config = valid_config();
        config.jupiter.api_key_env = " JUPITER_API_KEY".to_owned();
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.routes[0].forward_dexes = vec!["Raydium,Orca".to_owned()];
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.routes[0].return_dexes =
            vec!["Orca Whirlpool".to_owned(), "Orca Whirlpool".to_owned()];
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_unknown_configuration_fields() {
        let contents = r#"
            schema_version = 1

            [jupiter]
            base_url = "https://api.jup.ag/swap/v2"

            [scanner]
            min_profit_bps_typo = 30

            [[routes]]
            name = "USDC-WSOL-USDC"
            start_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            intermediate_mint = "So11111111111111111111111111111111111111112"
            amount = 100000000
        "#;

        assert!(toml::from_str::<Config>(contents).is_err());
    }
}
