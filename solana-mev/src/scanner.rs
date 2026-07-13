use std::collections::HashSet;

use anyhow::{Context, Result};

use crate::{
    config::{RouteConfig, ScannerConfig},
    jupiter::{JupiterClient, Quote, QuoteRequest},
};

pub struct Scanner {
    jupiter: JupiterClient,
    config: ScannerConfig,
    taker: String,
}

#[derive(Debug)]
pub struct Evaluation {
    pub route_name: String,
    pub start_amount: u64,
    pub intermediate_amount: u64,
    pub expected_final_amount: u64,
    pub minimum_final_amount: u64,
    pub estimated_net_profit: i128,
    pub estimated_profit_bps: i64,
    pub forward_venues: Vec<String>,
    pub return_venues: Vec<String>,
    pub venues_are_different: bool,
    pub is_opportunity: bool,
}

impl Scanner {
    pub fn new(jupiter: JupiterClient, config: ScannerConfig, taker: String) -> Self {
        Self {
            jupiter,
            config,
            taker,
        }
    }

    pub async fn evaluate(&self, route: &RouteConfig) -> Result<Evaluation> {
        let forward = self
            .jupiter
            .quote(self.quote_request(
                &route.start_mint,
                &route.intermediate_mint,
                route.amount,
                &route.forward_dexes,
            ))
            .await
            .with_context(|| format!("forward quote failed for {}", route.name))?;

        let backward = self
            .jupiter
            .quote(self.quote_request(
                &route.intermediate_mint,
                &route.start_mint,
                forward.out_amount,
                &route.return_dexes,
            ))
            .await
            .with_context(|| format!("return quote failed for {}", route.name))?;

        validate_leg_continuity(route, &forward, &backward)?;

        let estimated_net_profit = i128::from(backward.minimum_out_amount)
            - i128::from(route.amount)
            - i128::from(self.config.estimated_cost_in_start_units);
        let estimated_profit_bps = profit_bps(estimated_net_profit, route.amount);
        let venues_are_different =
            venue_sets_are_disjoint(&forward.venue_labels, &backward.venue_labels);
        let venue_requirement_met = !self.config.require_different_venues || venues_are_different;

        Ok(Evaluation {
            route_name: route.name.clone(),
            start_amount: route.amount,
            intermediate_amount: forward.out_amount,
            expected_final_amount: backward.out_amount,
            minimum_final_amount: backward.minimum_out_amount,
            estimated_net_profit,
            estimated_profit_bps,
            forward_venues: forward.venue_labels,
            return_venues: backward.venue_labels,
            venues_are_different,
            is_opportunity: estimated_profit_bps >= self.config.min_profit_bps
                && venue_requirement_met,
        })
    }

    fn quote_request<'a>(
        &'a self,
        input_mint: &'a str,
        output_mint: &'a str,
        amount: u64,
        dexes: &'a [String],
    ) -> QuoteRequest<'a> {
        QuoteRequest {
            input_mint,
            output_mint,
            amount,
            taker: &self.taker,
            slippage_bps: self.config.slippage_bps,
            max_accounts: self.config.max_accounts,
            fast_mode: self.config.fast_mode,
            dexes,
        }
    }
}

fn validate_leg_continuity(route: &RouteConfig, forward: &Quote, backward: &Quote) -> Result<()> {
    anyhow::ensure!(
        forward.input_mint == route.start_mint
            && forward.output_mint == route.intermediate_mint
            && backward.input_mint == route.intermediate_mint
            && backward.output_mint == route.start_mint
            && backward.in_amount == forward.out_amount,
        "Jupiter returned discontinuous route legs for {}",
        route.name
    );
    Ok(())
}

fn venue_sets_are_disjoint(forward: &[String], backward: &[String]) -> bool {
    if forward.is_empty() || backward.is_empty() {
        return false;
    }
    let forward: HashSet<_> = forward.iter().collect();
    backward.iter().all(|venue| !forward.contains(venue))
}

fn profit_bps(profit: i128, amount: u64) -> i64 {
    let bps = profit.saturating_mul(10_000) / i128::from(amount);
    bps.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

#[cfg(test)]
mod tests {
    use super::{profit_bps, venue_sets_are_disjoint};

    #[test]
    fn calculates_positive_and_negative_profit_bps() {
        assert_eq!(profit_bps(500_000, 100_000_000), 50);
        assert_eq!(profit_bps(-250_000, 100_000_000), -25);
    }

    #[test]
    fn requires_non_empty_disjoint_venue_sets() {
        assert!(venue_sets_are_disjoint(
            &["Raydium CLMM".to_owned()],
            &["Orca Whirlpool".to_owned()]
        ));
        assert!(!venue_sets_are_disjoint(
            &["Raydium CLMM".to_owned()],
            &["Raydium CLMM".to_owned()]
        ));
        assert!(!venue_sets_are_disjoint(
            &[],
            &["Orca Whirlpool".to_owned()]
        ));
    }
}
