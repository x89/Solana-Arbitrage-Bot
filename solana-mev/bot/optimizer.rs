use crate::models::mev_opportunity::MevOpportunity;
use solana_client::rpc_client::RpcClient;
use solana_sdk::transaction::Transaction;

pub struct Optimizer {
    rpc_client: RpcClient,
}

impl Optimizer {
    pub fn new(rpc_client: RpcClient) -> Self {
        Self { rpc_client }
    }

    pub async fn optimize(&self, opportunity: &MevOpportunity) -> Vec<Transaction> {
        let mut optimized_txs = Vec::new();
        for (tx, profit) in &opportunity.transactions {
            if *profit >= opportunity.min_profit {
                optimized_txs.push(tx.clone());
            }
        }
        optimized_txs
    }
}