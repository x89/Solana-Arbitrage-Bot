use std::{collections::HashSet, time::Duration};

use anyhow::{Context, Result};
use tokio::time::Instant;

use crate::{
    config::{RouteConfig, ScannerConfig},
    jupiter::{JupiterClient, Quote, QuoteRequest},
};

#[allow(async_fn_in_trait)]
pub trait QuoteProvider {
    async fn quote(&self, request: QuoteRequest<'_>) -> Result<Quote>;
}

impl QuoteProvider for JupiterClient {
    async fn quote(&self, request: QuoteRequest<'_>) -> Result<Quote> {
        JupiterClient::quote(self, request).await
    }
}

pub struct Scanner<Q = JupiterClient> {
    jupiter: Q,
    config: ScannerConfig,
    taker: String,
}

#[derive(Debug)]
pub struct Evaluation {
    pub route_name: String,
    pub start_amount: u64,
    pub expected_intermediate_amount: u64,
    pub minimum_intermediate_amount: u64,
    pub expected_final_amount: u64,
    pub minimum_final_amount: u64,
    pub estimated_net_profit: i128,
    pub estimated_profit_bps: i64,
    pub forward_venues: Vec<String>,
    pub return_venues: Vec<String>,
    pub venues_are_different: bool,
    pub cycle_duration_ms: u64,
    pub is_opportunity: bool,
}

impl<Q: QuoteProvider> Scanner<Q> {
    pub fn new(jupiter: Q, config: ScannerConfig, taker: String) -> Self {
        Self {
            jupiter,
            config,
            taker,
        }
    }

    pub async fn evaluate(&self, route: &RouteConfig) -> Result<Evaluation> {
        let started = Instant::now();
        let maximum_duration = Duration::from_millis(self.config.max_cycle_duration_ms);
        let mut evaluation = tokio::time::timeout(maximum_duration, self.evaluate_inner(route))
            .await
            .with_context(|| {
                format!(
                    "quote cycle for {} exceeded {} ms",
                    route.name, self.config.max_cycle_duration_ms
                )
            })??;
        evaluation.cycle_duration_ms = started.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
        Ok(evaluation)
    }

    async fn evaluate_inner(&self, route: &RouteConfig) -> Result<Evaluation> {
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
                forward.minimum_out_amount,
                &route.return_dexes,
            ))
            .await
            .with_context(|| format!("return quote failed for {}", route.name))?;

        validate_leg_continuity(route, &forward, &backward)?;

        let estimated_net_profit = i128::from(backward.minimum_out_amount)
            - i128::from(route.amount)
            - i128::from(route.estimated_cost_in_start_units);
        let estimated_profit_bps = profit_bps(estimated_net_profit, route.amount);
        let venues_are_different = venue_sets_are_disjoint(&forward.amm_keys, &backward.amm_keys);
        let venue_requirement_met = !self.config.require_different_venues || venues_are_different;

        Ok(Evaluation {
            route_name: route.name.clone(),
            start_amount: route.amount,
            expected_intermediate_amount: forward.out_amount,
            minimum_intermediate_amount: forward.minimum_out_amount,
            expected_final_amount: backward.out_amount,
            minimum_final_amount: backward.minimum_out_amount,
            estimated_net_profit,
            estimated_profit_bps,
            forward_venues: forward.venue_labels,
            return_venues: backward.venue_labels,
            venues_are_different,
            cycle_duration_ms: 0,
            is_opportunity: meets_profit_threshold(
                estimated_net_profit,
                route.amount,
                self.config.min_profit_bps,
            ) && venue_requirement_met,
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
            && backward.in_amount == forward.minimum_out_amount,
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

fn meets_profit_threshold(profit: i128, amount: u64, minimum_bps: i64) -> bool {
    profit > 0
        && profit.saturating_mul(10_000)
            >= i128::from(amount).saturating_mul(i128::from(minimum_bps))
}

#[cfg(test)]
mod tests {
    use super::{meets_profit_threshold, profit_bps, venue_sets_are_disjoint};

    #[test]
    fn calculates_positive_and_negative_profit_bps() {
        assert_eq!(profit_bps(500_000, 100_000_000), 50);
        assert_eq!(profit_bps(-250_000, 100_000_000), -25);
    }

    #[test]
    fn requires_positive_profit_and_compares_without_rounding() {
        assert!(!meets_profit_threshold(-1, 100_000_000, 0));
        assert!(!meets_profit_threshold(0, 100_000_000, 0));
        assert!(meets_profit_threshold(1, 100_000_000, 0));
        assert!(!meets_profit_threshold(9_999, 100_000_000, 1));
        assert!(meets_profit_threshold(10_000, 100_000_000, 1));
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
