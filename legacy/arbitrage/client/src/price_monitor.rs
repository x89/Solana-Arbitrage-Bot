use {
    crate::types::{PriceData, DexType},
    crate::dex_client::DexClient,
    std::{sync::Arc, collections::HashMap},
    tokio::sync::RwLock,
    async_trait::async_trait,
};

pub struct PriceMonitor {
    dex_clients: Vec<Arc<dyn DexClient>>,
    price_cache: Arc<RwLock<HashMap<String, PriceData>>>,
}

impl PriceMonitor {
    pub fn new(dex_clients: Vec<Arc<dyn DexClient>>) -> Self {
        Self {
            dex_clients,
            price_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn start_monitoring(&self) -> Result<(), Box<dyn std::error::Error>> {
        let price_cache = self.price_cache.clone();
        
        for dex_client in &self.dex_clients {
            let dex_client = dex_client.clone();
            let price_cache = price_cache.clone();
            
            tokio::spawn(async move {
                loop {
                    if let Ok(prices) = dex_client.get_prices().await {
                        let mut cache = price_cache.write().await;
                        for price in prices {
                            cache.insert(
                                format!("{}:{}:{}", 
                                    price.token_in, 
                                    price.token_out,
                                    price.dex_type
                                ),
                                price
                            );
                        }
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                }
            });
        }

        Ok(())
    }

    pub async fn get_latest_prices(&self) -> Result<Vec<PriceData>, Box<dyn std::error::Error>> {
        let cache = self.price_cache.read().await;
        Ok(cache.values().cloned().collect())
    }

    pub async fn get_price(
        &self,
        token_in: &str,
        token_out: &str,
        dex_type: DexType
    ) -> Option<PriceData> {
        let cache = self.price_cache.read().await;
        cache.get(&format!("{}:{}:{}", token_in, token_out, dex_type))
            .cloned()
    }
} 