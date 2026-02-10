use solana_client::rpc_client::RpcClient;
use solana_sdk::transaction::Transaction;

pub struct RiskManager {
    rpc_client: RpcClient,
    max_capital_per_trade: f64,
    max_slippage: f64,
}

impl RiskManager {
    pub fn new(rpc_client: RpcClient) -> Self {
        Self {
            rpc_client,
            max_capital_per_trade: 1000.0,
            max_slippage: 0.05,
        }
    }

    pub fn update(&mut self, max_capital_per_trade: f64, max_slippage: f64) {
        self.max_capital_per_trade = max_capital_per_trade;
        self.max_slippage = max_slippage;
    }

    pub async fn is_safe(&self, tx: &Transaction) -> bool {
        let tx_cost = self.calculate_tx_cost(tx).await;
        tx_cost <= self.max_capital_per_trade && self.calculate_slippage(tx).await <= self.max_slippage
    }

    async fn calculate_tx_cost(&self, tx: &Transaction) -> f64 {
        let (result, _) = self.rpc_client.simulate_transaction(tx).await.unwrap();
        let accounts_data = result.accounts.unwrap();
        let mut cost = 0.0;
        for account in &accounts_data {
            let lamports = account.lamports.unwrap();
            cost += lamports as f64 / 1e9;
        }
        cost
    }

    async fn calculate_slippage(&self, tx: &Transaction) -> f64 {
        let (result, _) = self.rpc_client.simulate_transaction(tx).await.unwrap();
        let accounts_data = result.accounts.unwrap();
        let mut balance_changes = Vec::new();
        for account in &accounts_data {
            let lamports = account.lamports.unwrap();
            balance_changes.push(lamports as f64 / 1e9);
        }
        let max_balance_change = balance_changes.iter().cloned().fold(0.0 / 0.0, f64::max);
        let min_balance_change = balance_changes.iter().cloned().fold(0.0 / 0.0, f64::min);
        (max_balance_change - min_balance_change) / min_balance_change
    }
}