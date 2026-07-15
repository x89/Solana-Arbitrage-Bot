use solana_sdk::transaction::Transaction;

pub struct GasOptimizer {}

impl GasOptimizer {
    pub fn new(_rpc_client: solana_client::rpc_client::RpcClient) -> Self {
        Self {}
    }

    pub fn update(&mut self, _market_conditions: &crate::models::market_conditions::MarketConditions) {}

    pub async fn optimize(&self, txs: &[Transaction]) -> Vec<Transaction> {
        let optimized_txs = txs.to_vec();
        optimized_txs
    }
}