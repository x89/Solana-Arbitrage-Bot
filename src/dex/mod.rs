use anyhow::Result;
use chrono::{DateTime, Local};
use async_trait::async_trait;
use solana_client::rpc_client::RpcClient;
use std::sync::Arc;
use serde::{Serialize, Deserialize};

pub mod raydium;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPrice {
    pub token_address: String,
    pub dex_name: String,
    pub price: f64,
    pub timestamp: DateTime<Local>,
}

#[async_trait]
pub trait DexProtocol: Send + Sync {
    fn name(&self) -> &str;
    fn clone_box(&self) -> Box<dyn DexProtocol + Send + Sync>;
    async fn get_token_price(&self, rpc_client: Arc<RpcClient>, token_mint: &str) -> Result<Option<f64>>;
}

impl Clone for Box<dyn DexProtocol + Send + Sync> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
} 