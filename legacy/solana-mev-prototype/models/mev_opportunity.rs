use solana_sdk::transaction::Transaction;

pub struct MevOpportunity {
    pub transactions: Vec<(Transaction, f64)>,
    pub min_profit: f64,
}