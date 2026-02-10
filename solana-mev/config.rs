use serde::Deserialize;
use solana_sdk::pubkey::Pubkey;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub solana: SolanaConfig,
    pub bot: BotConfig,
    pub dexes: DexesConfig,
    pub monitoring: MonitoringConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SolanaConfig {
    pub rpc_url: String,
    pub ws_url: String,
    pub commitment: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct BotConfig {
    pub keypair_path: String,
    pub profit_threshold: f64,
    pub max_position_size: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DexesConfig {
    pub raydium_program_id: Pubkey,
    pub serum_program_id: Pubkey,
    pub orca_program_id: Pubkey,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MonitoringConfig {
    pub dashboard_port: u16,
    pub update_interval: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LoggingConfig {
    pub level: String,
}