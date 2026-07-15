use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{anyhow, Result};
use mev_bot_solana::{
    config::{RouteConfig, ScannerConfig},
    jupiter::{Quote, QuoteRequest},
    scanner::{QuoteProvider, Scanner},
};

const START_MINT: &str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";
const INTERMEDIATE_MINT: &str = "So11111111111111111111111111111111111111112";

struct FakeQuoteProvider {
    quotes: Mutex<VecDeque<Quote>>,
    requested_amounts: Arc<Mutex<Vec<u64>>>,
    delay: Duration,
}

impl FakeQuoteProvider {
    fn new(quotes: impl IntoIterator<Item = Quote>) -> Self {
        Self {
            quotes: Mutex::new(quotes.into_iter().collect()),
            requested_amounts: Arc::new(Mutex::new(Vec::new())),
            delay: Duration::ZERO,
        }
    }

    fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }
}

impl QuoteProvider for FakeQuoteProvider {
    async fn quote(&self, request: QuoteRequest<'_>) -> Result<Quote> {
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }
        self.requested_amounts
            .lock()
            .expect("request lock")
            .push(request.amount);
        self.quotes
            .lock()
            .expect("quote lock")
            .pop_front()
            .ok_or_else(|| anyhow!("fake quote queue exhausted"))
    }
}

fn scanner_config() -> ScannerConfig {
    ScannerConfig {
        interval_ms: 1_000,
        min_profit_bps: 30,
        slippage_bps: 30,
        max_accounts: 64,
        fast_mode: true,
        require_different_venues: true,
        max_cycle_duration_ms: 1_000,
    }
}

fn route() -> RouteConfig {
    RouteConfig {
        name: "USDC-WSOL-USDC".to_owned(),
        start_mint: START_MINT.to_owned(),
        intermediate_mint: INTERMEDIATE_MINT.to_owned(),
        amount: 100_000_000,
        estimated_cost_in_start_units: 10_000,
        forward_dexes: Vec::new(),
        return_dexes: Vec::new(),
        enabled: true,
    }
}

fn quote(
    input_mint: &str,
    output_mint: &str,
    in_amount: u64,
    out_amount: u64,
    minimum_out_amount: u64,
    pool: &str,
) -> Quote {
    Quote {
        input_mint: input_mint.to_owned(),
        output_mint: output_mint.to_owned(),
        in_amount,
        out_amount,
        minimum_out_amount,
        venue_labels: vec![format!("venue-{pool}")],
        amm_keys: vec![pool.to_owned()],
    }
}

#[tokio::test]
async fn chains_from_forward_minimum_and_applies_costs() {
    let provider = FakeQuoteProvider::new([
        quote(
            START_MINT,
            INTERMEDIATE_MINT,
            100_000_000,
            75_000_000,
            70_000_000,
            "pool-a",
        ),
        quote(
            INTERMEDIATE_MINT,
            START_MINT,
            70_000_000,
            101_000_000,
            100_500_000,
            "pool-b",
        ),
    ]);
    let requested_amounts = Arc::clone(&provider.requested_amounts);
    let scanner = Scanner::new(provider, scanner_config(), "unused-taker".to_owned());

    let evaluation = scanner.evaluate(&route()).await.unwrap();

    assert_eq!(evaluation.minimum_intermediate_amount, 70_000_000);
    assert_eq!(evaluation.estimated_net_profit, 490_000);
    assert_eq!(evaluation.estimated_profit_bps, 49);
    assert!(evaluation.venues_are_different);
    assert!(evaluation.is_opportunity);
    assert_eq!(
        requested_amounts.lock().expect("request lock").as_slice(),
        [100_000_000, 70_000_000]
    );
}

#[tokio::test]
async fn rejects_shared_pool_and_stale_quote_cycle() {
    let shared_pool_provider = FakeQuoteProvider::new([
        quote(
            START_MINT,
            INTERMEDIATE_MINT,
            100_000_000,
            75_000_000,
            70_000_000,
            "shared-pool",
        ),
        quote(
            INTERMEDIATE_MINT,
            START_MINT,
            70_000_000,
            101_000_000,
            100_500_000,
            "shared-pool",
        ),
    ]);
    let scanner = Scanner::new(
        shared_pool_provider,
        scanner_config(),
        "unused-taker".to_owned(),
    );
    let evaluation = scanner.evaluate(&route()).await.unwrap();
    assert!(!evaluation.venues_are_different);
    assert!(!evaluation.is_opportunity);

    let delayed_provider = FakeQuoteProvider::new([quote(
        START_MINT,
        INTERMEDIATE_MINT,
        100_000_000,
        75_000_000,
        70_000_000,
        "pool-a",
    )])
    .with_delay(Duration::from_millis(50));
    let mut config = scanner_config();
    config.max_cycle_duration_ms = 10;
    let scanner = Scanner::new(delayed_provider, config, "unused-taker".to_owned());

    let error = scanner.evaluate(&route()).await.unwrap_err();
    assert!(error.to_string().contains("exceeded 10 ms"));
}
