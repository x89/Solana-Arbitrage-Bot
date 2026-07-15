use crate::dex::dex_manager::DexManager;
use crate::error::Result;
use crate::models::market::Market;
use crate::models::sniping_opportunity::SnipingOpportunity;
use solana_client::rpc_client::RpcClient;
use solana_sdk::pubkey::Pubkey;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct SnipingStrategy {
    pub rpc_client: Arc<RpcClient>,
    pub dex_manager: Arc<Mutex<DexManager>>,
    pub target_markets: Vec<Pubkey>,
    pub max_price: f64,
    pub min_liquidity: f64,
}

impl SnipingStrategy {
    pub fn new(
        rpc_client: Arc<RpcClient>,
        dex_manager: Arc<Mutex<DexManager>>,
        target_markets: Vec<Pubkey>,
        max_price: f64,
        min_liquidity: f64,
    ) -> Self {
        SnipingStrategy {
            rpc_client,
            dex_manager,
            target_markets,
            max_price,
            min_liquidity,
        }
    }

    pub async fn run(&self) -> Result<()> {
        loop {
            let markets = self.get_target_markets().await?;
            
            let opportunities = self.find_opportunities(&markets).await?;
            
            for opportunity in opportunities {
                self.execute_sniping(&opportunity).await?;
            }
            
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }

    async fn get_target_markets(&self) -> Result<Vec<Market>> {
        let mut markets = Vec::new();
        
        for market_address in &self.target_markets {
            let market = self.dex_manager.lock().await.get_market(market_address).await?;
            markets.push(market);
        }
        
        Ok(markets)
    }

    async fn find_opportunities(&self, markets: &[Market]) -> Result<Vec<SnipingOpportunity>> {
        let mut opportunities = Vec::new();
        
        for market in markets {
            let orderbook = self.dex_manager.lock().await.get_orderbook(market).await?;
            
            if let Some(best_bid) = orderbook.bids.first() {
                if best_bid.price <= self.max_price {
                    let liquidity = self.calculate_liquidity(market, &orderbook).await?;
                    
                    if liquidity >= self.min_liquidity {
                        let opportunity = SnipingOpportunity {
                            market: market.clone(),
                            price: best_bid.price,
                            liquidity,
                        };
                        opportunities.push(opportunity);
                    }
                }
            }
        }
        
        Ok(opportunities)
    }

    async fn calculate_liquidity(&self, market: &Market, orderbook: &Orderbook) -> Result<f64> {
        let bids_volume = orderbook.bids.iter().map(|order| order.quantity).sum();
        let asks_volume = orderbook.asks.iter().map(|order| order.quantity).sum();
        
        let mid_price = (orderbook.bids[0].price + orderbook.asks[0].price) / 2.0;
        
        let base_volume = bids_volume + asks_volume;
        let quote_volume = base_volume * mid_price;
        
        let base_decimals = market.base_decimals;
        let quote_decimals = market.quote_decimals;
        
        let liquidity = base_volume / 10_usize.pow(base_decimals as u32) as f64
            + quote_volume / 10_usize.pow(quote_decimals as u32) as f64;
        
        Ok(liquidity)
    }

    async fn execute_sniping(&self, opportunity: &SnipingOpportunity) -> Result<()> {
        let market = &opportunity.market;
        let price = opportunity.price;
        let quantity = opportunity.liquidity / price;

        let order = self
            .dex_manager
            .lock()
            .await
            .place_order(market, OrderType::Limit, OrderSide::Bid, price, quantity)
            .await?;

        println!("Placed sniping order: {:?}", order);

        Ok(())
    }
}