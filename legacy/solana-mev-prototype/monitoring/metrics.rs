use crate::models::market::Market;
use crate::models::order::Order;
use solana_sdk::pubkey::Pubkey;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct Metrics {
    pub orders: Arc<Mutex<Vec<Order>>>,
    pub profits: Arc<Mutex<HashMap<Pubkey, f64>>>,
    pub volumes: Arc<Mutex<HashMap<Market, f64>>>,
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            orders: Arc::new(Mutex::new(Vec::new())),
            profits: Arc::new(Mutex::new(HashMap::new())),
            volumes: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn add_order(&self, order: Order) {
        self.orders.lock().await.push(order);
    }

    pub async fn update_profit(&self, market: &Market, profit: f64) {
        let mut profits = self.profits.lock().await;
        *profits.entry(market.address).or_insert(0.0) += profit;
    }

    pub async fn update_volume(&self, market: &Market, volume: f64) {
        let mut volumes = self.volumes.lock().await;
        *volumes.entry(market.clone()).or_insert(0.0) += volume;
    }

    pub async fn get_orders(&self) -> Vec<Order> {
        self.orders.lock().await.clone()
    }

    pub async fn get_profits(&self) -> HashMap<Pubkey, f64> {
        self.profits.lock().await.clone()
    }

    pub async fn get_volumes(&self) -> HashMap<Market, f64> {
        self.volumes.lock().await.clone()
    }
}