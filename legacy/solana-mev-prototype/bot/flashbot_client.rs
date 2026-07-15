use crate::utils::solana::send_transaction;
use solana_client::rpc_client::RpcClient;
use solana_sdk::transaction::Transaction;

pub struct FlashbotsClient {
    rpc_client: RpcClient,
}

impl FlashbotsClient {
    pub fn new(rpc_client: RpcClient) -> Self {
        Self { rpc_client }
    }

    pub async fn send_bundle(&self, txs: &[Transaction]) -> Result<(), Box<dyn std::error::Error>> {
        for tx in txs {
            send_transaction(&self.rpc_client, tx).await?;
        }
        Ok(())
    }
}