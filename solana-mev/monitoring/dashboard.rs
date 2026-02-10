use crate::monitoring::metrics::Metrics;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time;

pub struct Dashboard {
    pub metrics: Arc<Metrics>,
    pub update_interval: u64,
}

impl Dashboard {
    pub fn new(metrics: Arc<Metrics>, update_interval: u64) -> Self {
        Dashboard {
            metrics,
            update_interval,
        }
    }

    pub async fn run(&self) {
        loop {
            self.render().await;
            time::sleep(time::Duration::from_secs(self.update_interval)).await;
        }
    }

    async fn render(&self) {
        let orders = self.metrics.get_orders().await;
        let profits = self.metrics.get_profits().await;
        let volumes = self.metrics.get_volumes().await;
        
        println!("=== MEV Bot Dashboard ===");
        println!("Total Orders: {}", orders.len());
        println!("Total Profit: {:.2} SOL", profits.values().sum::<f64>());
        println!("Total Volume: {:.2} SOL", volumes.values().sum::<f64>());
        println!("==========================");
    }
}