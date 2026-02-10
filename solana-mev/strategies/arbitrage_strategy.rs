use crate::dex::dex_manager::DexManager;
use crate::models::market_conditions::MarketConditions;
use crate::models::arbitrage_opportunity::ArbitrageOpportunity;
use crate::strategies::strategy::Strategy;
use crate::utils::math;
use async_trait::async_trait;
use solana_client::rpc_client::RpcClient;
use solana_sdk::pubkey::Pubkey;
use std::collections::HashMap;

pub struct ArbitrageStrategy {
    pub rpc_client: RpcClient,
    pub dex_manager: DexManager,
    pub min_profit_threshold: f64,
}

impl ArbitrageStrategy {
    pub fn new(rpc_client: RpcClient, dex_manager: DexManager, min_profit_threshold: f64) -> Self {
        ArbitrageStrategy {
            rpc_client,
            dex_manager,
            min_profit_threshold,
        }
    }

    pub async fn find_arbitrage_opportunities(&self, market_conditions: &MarketConditions) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();

        let token_prices = market_conditions.token_prices.clone();
        let token_pairs = self.generate_token_pairs(&token_prices);

        for (token_a, token_b) in token_pairs {
            if let Some(opportunity) = self.find_arbitrage_opportunity(token_a, token_b, &token_prices).await {
                if opportunity.expected_profit >= self.min_profit_threshold {
                    opportunities.push(opportunity);
                }
            }
        }

        opportunities
    }

    async fn find_arbitrage_opportunity(&self, token_a: &str, token_b: &str, token_prices: &HashMap<String, f64>) -> Option<ArbitrageOpportunity> {
        let mut best_opportunity: Option<ArbitrageOpportunity> = None;

        if let Some(price_a_b) = token_prices.get(&format!("{}/{}", token_a, token_b)) {
            if let Some(price_b_a) = token_prices.get(&format!("{}/{}", token_b, token_a)) {
                let forward_amount = 1.0;
                let forward_price = price_a_b;
                let backward_amount = forward_amount * forward_price;
                let backward_price = price_b_a;

                let forward_trade = self.dex_manager.get_best_trade_route(token_a, token_b, forward_amount).await;
                let backward_trade = self.dex_manager.get_best_trade_route(token_b, token_a, backward_amount).await;

                if let (Some(forward_trade), Some(backward_trade)) = (forward_trade, backward_trade) {
                    let forward_amount_received = math::checked_div(forward_amount, forward_trade.price).unwrap_or(0.0);
                    let backward_amount_received = math::checked_mul(backward_trade.received_amount, backward_price).unwrap_or(0.0);

                    let expected_profit = backward_amount_received - forward_amount;

                    if expected_profit > 0.0 {
                        best_opportunity = Some(ArbitrageOpportunity {
                            token_a: token_a.to_string(),
                            token_b: token_b.to_string(),
                            forward_trade,
                            backward_trade,
                            expected_profit,
                        });
                    }
                }
            }
        }

        best_opportunity
    }

    fn generate_token_pairs(&self, token_prices: &HashMap<String, f64>) -> Vec<(String, String)> {
        let mut pairs = Vec::new();

        for (token_a, _) in token_prices {
            for (token_b, _) in token_prices {
                if token_a != token_b {
                    pairs.push((token_a.clone(), token_b.clone()));
                }
            }
        }

        pairs
    }
}

#[async_trait]
impl Strategy for ArbitrageStrategy {
    async fn find_opportunities(&self, market_conditions: &MarketConditions) -> Vec<ArbitrageOpportunity> {
        self.find_arbitrage_opportunities(market_conditions).await
    }

    async fn execute_opportunities(&self, opportunities: &[ArbitrageOpportunity]) {
        for opportunity in opportunities {
            let forward_trade = &opportunity.forward_trade;
            let backward_trade = &opportunity.backward_trade;

            let forward_result = self.dex_manager.execute_trade(forward_trade).await;
            if forward_result.is_ok() {
                let backward_result = self.dex_manager.execute_trade(backward_trade).await;
                if backward_result.is_ok() {
                    // Log successful arbitrage execution
                } else {
                    // Log backward trade failure
                }
            } else {
                // Log forward trade failure
            }
        }
    }
}