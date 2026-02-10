use solana_sdk::transaction::Transaction;

pub struct TradeExecutor {}

impl TradeExecutor {
    pub fn new(_rpc_client: solana_client::rpc_client::RpcClient) -> Self {
        Self {}
    }

    pub async fn execute_transactions(&self, txs: &[Transaction]) -> Vec<Transaction> {
        let executed_txs = txs.to_vec();
        executed_txs
    }
}