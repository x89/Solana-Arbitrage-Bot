use solana_sdk::transaction::Transaction;

pub struct OrderManager {}

impl OrderManager {
    pub fn new(_rpc_client: solana_client::rpc_client::RpcClient) -> Self {
        Self {}
    }

    pub fn update(&mut self, _market_conditions: &crate::models::market_conditions::MarketConditions) {}

    pub async fn manage_orders(&self, _executed_txs: &[Transaction]) {}
}