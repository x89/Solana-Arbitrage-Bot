use {
    crate::types::{PriceData, ArbitrageOpportunity, RouteType, DexType},
    std::collections::HashMap,
};

pub struct RouteFinder {
    min_profit_threshold: f64,
    max_slippage: f64,
}

impl RouteFinder {
    pub fn new(min_profit_threshold: f64, max_slippage: f64) -> Self {
        Self {
            min_profit_threshold,
            max_slippage,
        }
    }

    pub async fn find_opportunities(
        &self,
        prices: &[PriceData],
    ) -> Result<Vec<ArbitrageOpportunity>, Box<dyn std::error::Error>> {
        let mut opportunities = Vec::new();

        // Build price map for quick lookup
        let price_map: HashMap<_, _> = prices.iter()
            .map(|p| ((p.token_in.clone(), p.token_out.clone(), p.dex_type), p))
            .collect();

        // Find two-hop opportunities (Meteora -> Raydium)
        opportunities.extend(self.find_two_hop_opportunities(&price_map));

        // Find triangle opportunities (Meteora -> Meteora -> Raydium)
        opportunities.extend(self.find_triangle_opportunities(&price_map));

        // Find Orca-Whirlpool opportunities
        opportunities.extend(self.find_orca_whirlpool_opportunities(&price_map));

        Ok(opportunities)
    }

    fn find_two_hop_opportunities(
        &self,
        price_map: &HashMap<(String, String, DexType), &PriceData>,
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();

        for (key1, price1) in price_map.iter() {
            if price1.dex_type != DexType::Meteora {
                continue;
            }

            // Look for matching Raydium route
            if let Some(price2) = price_map.get(&(
                price1.token_out.clone(),
                price1.token_in.clone(),
                DexType::Raydium,
            )) {
                let amount_out = self.calculate_output_amount(
                    price1.amount_in,
                    &[price1.clone(), price2.clone()],
                );

                if let Some(final_amount) = amount_out {
                    let profit_ratio = (final_amount - price1.amount_in) / price1.amount_in;
                    
                    if profit_ratio > self.min_profit_threshold {
                        opportunities.push(ArbitrageOpportunity {
                            route_type: RouteType::TwoHop,
                            amount_in: price1.amount_in,
                            token_in: price1.token_in.clone(),
                            intermediate_amounts: vec![price1.amount_out],
                            expected_output: final_amount,
                            profit_ratio,
                            steps: vec![price1.clone(), price2.clone()],
                        });
                    }
                }
            }
        }

        opportunities
    }

    fn find_triangle_opportunities(
        &self,
        price_map: &HashMap<(String, String, DexType), &PriceData>,
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();

        for (key1, price1) in price_map.iter() {
            if price1.dex_type != DexType::Meteora {
                continue;
            }

            // Look for second Meteora hop
            if let Some(price2) = price_map.get(&(
                price1.token_out.clone(),
                format!("intermediate_token"),
                DexType::Meteora,
            )) {
                // Look for final Raydium hop
                if let Some(price3) = price_map.get(&(
                    format!("intermediate_token"),
                    price1.token_in.clone(),
                    DexType::Raydium,
                )) {
                    let amount_out = self.calculate_output_amount(
                        price1.amount_in,
                        &[price1.clone(), price2.clone(), price3.clone()],
                    );

                    if let Some(final_amount) = amount_out {
                        let profit_ratio = (final_amount - price1.amount_in) / price1.amount_in;
                        
                        if profit_ratio > self.min_profit_threshold {
                            opportunities.push(ArbitrageOpportunity {
                                route_type: RouteType::Triangle,
                                amount_in: price1.amount_in,
                                token_in: price1.token_in.clone(),
                                intermediate_amounts: vec![price1.amount_out, price2.amount_out],
                                expected_output: final_amount,
                                profit_ratio,
                                steps: vec![price1.clone(), price2.clone(), price3.clone()],
                            });
                        }
                    }
                }
            }
        }

        opportunities
    }

    fn find_orca_whirlpool_opportunities(
        &self,
        price_map: &HashMap<(String, String, DexType), &PriceData>,
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();

        for (key1, price1) in price_map.iter() {
            if price1.dex_type != DexType::Orca {
                continue;
            }

            // Look for Whirlpool route
            if let Some(price2) = price_map.get(&(
                price1.token_out.clone(),
                price1.token_in.clone(),
                DexType::Whirlpool,
            )) {
                let amount_out = self.calculate_output_amount(
                    price1.amount_in,
                    &[price1.clone(), price2.clone()],
                );

                if let Some(final_amount) = amount_out {
                    let profit_ratio = (final_amount - price1.amount_in) / price1.amount_in;
                    
                    if profit_ratio > self.min_profit_threshold {
                        opportunities.push(ArbitrageOpportunity {
                            route_type: RouteType::OrcaWhirlpool,
                            amount_in: price1.amount_in,
                            token_in: price1.token_in.clone(),
                            intermediate_amounts: vec![price1.amount_out],
                            expected_output: final_amount,
                            profit_ratio,
                            steps: vec![price1.clone(), price2.clone()],
                        });
                    }
                }
            }
        }

        opportunities
    }

    fn calculate_output_amount(&self, amount_in: f64, steps: &[PriceData]) -> Option<f64> {
        let mut current_amount = amount_in;

        for step in steps {
            // Apply slippage and fees
            let slippage = 1.0 - self.max_slippage;
            let fee_multiplier = match step.dex_type {
                DexType::Meteora => 0.998, // 0.2% fee
                DexType::Raydium => 0.9975, // 0.25% fee
                DexType::Orca => 0.997, // 0.3% fee
                DexType::Whirlpool => 0.997, // 0.3% fee
            };

            current_amount = current_amount * step.price * fee_multiplier * slippage;
        }

        Some(current_amount)
    }
} 