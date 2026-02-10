use solana_client::rpc_client::RpcClient;
use solana_sdk::transaction::Transaction;

pub struct SimulationEngine {
    rpc_client: RpcClient,
}

impl SimulationEngine {
    pub fn new(rpc_client: RpcClient) -> Self {
        Self { rpc_client }
    }

    pub async fn simulate(&self, tx: &Transaction) -> f64 {
        let (result, _) = self.rpc_client.simulate_transaction(tx).await.unwrap();
        let accounts_data = result.accounts.unwrap();
        let mut profit = 0.0;
        for account in &accounts_data {
            let lamports = account.lamports.unwrap();
            profit += lamports as f64 / 1e9;
        }
        profit
    }
}