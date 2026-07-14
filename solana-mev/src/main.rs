use std::{path::PathBuf, time::Duration};

use anyhow::{bail, Result};
use clap::Parser;
use mev_bot_solana::{
    config::{Config, RouteConfig},
    jupiter::{api_error_details, JupiterClient},
    scanner::Scanner,
};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "solana-arbitrage-monitor",
    version,
    about = "Dry-run Solana cyclic-arbitrage monitor using Jupiter Swap V2"
)]
struct Args {
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// Scan once and exit instead of monitoring continuously.
    #[arg(long)]
    once: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    let config = Config::load(&args.config)?;
    let api_key = config.api_key()?;
    let taker = config.taker()?;
    let jupiter = JupiterClient::new(
        &config.jupiter.base_url,
        api_key,
        Duration::from_millis(config.jupiter.request_timeout_ms),
        Duration::from_millis(config.jupiter.min_request_interval_ms),
    )?;
    let scanner = Scanner::new(jupiter, config.scanner.clone(), taker);
    let routes: Vec<_> = config
        .routes
        .into_iter()
        .filter(|route| route.enabled)
        .collect();

    info!(
        routes = routes.len(),
        api = %config.jupiter.base_url,
        "starting dry-run monitor; transaction execution is intentionally disabled"
    );

    let mut consecutive_failed_scans = 0_u32;
    loop {
        let outcome = scan_once(&scanner, &routes).await;
        if outcome.permanent_error {
            bail!("Jupiter rejected the configured credentials; monitoring stopped");
        }
        if args.once {
            anyhow::ensure!(
                outcome.errors == 0,
                "{} route evaluation(s) failed",
                outcome.errors
            );
            break;
        }

        let base_interval = Duration::from_millis(config.scanner.interval_ms);
        let delay = if outcome.errors == routes.len() {
            consecutive_failed_scans = consecutive_failed_scans.saturating_add(1);
            let multiplier = 1_u32 << consecutive_failed_scans.min(6);
            base_interval
                .saturating_mul(multiplier)
                .min(Duration::from_secs(60))
        } else {
            consecutive_failed_scans = 0;
            base_interval
        };
        let delay = delay.max(
            outcome
                .retry_after
                .unwrap_or_default()
                .min(Duration::from_secs(300)),
        );
        if outcome.errors > 0 {
            warn!(
                errors = outcome.errors,
                retry_in_ms = delay.as_millis(),
                "scan completed with errors; applying backoff"
            );
        }

        tokio::select! {
            _ = tokio::time::sleep(delay) => {}
            signal = tokio::signal::ctrl_c() => {
                signal?;
                info!("shutdown requested");
                break;
            }
        }
    }

    Ok(())
}

#[derive(Default)]
struct ScanOutcome {
    errors: usize,
    permanent_error: bool,
    retry_after: Option<Duration>,
}

async fn scan_once(scanner: &Scanner, routes: &[RouteConfig]) -> ScanOutcome {
    let mut outcome = ScanOutcome::default();
    for route in routes {
        match scanner.evaluate(route).await {
            Ok(evaluation) if evaluation.is_opportunity => {
                warn!(
                    route = %evaluation.route_name,
                    start_amount = evaluation.start_amount,
                    expected_intermediate_amount = evaluation.expected_intermediate_amount,
                    minimum_intermediate_amount = evaluation.minimum_intermediate_amount,
                    expected_final_amount = evaluation.expected_final_amount,
                    minimum_final_amount = evaluation.minimum_final_amount,
                    estimated_net_profit = evaluation.estimated_net_profit,
                    estimated_profit_bps = evaluation.estimated_profit_bps,
                    forward_venues = ?evaluation.forward_venues,
                    return_venues = ?evaluation.return_venues,
                    "candidate opportunity detected; this is not an execution guarantee"
                );
            }
            Ok(evaluation) => {
                info!(
                    route = %evaluation.route_name,
                    estimated_profit_bps = evaluation.estimated_profit_bps,
                    minimum_final_amount = evaluation.minimum_final_amount,
                    forward_venues = ?evaluation.forward_venues,
                    return_venues = ?evaluation.return_venues,
                    venues_are_different = evaluation.venues_are_different,
                    "route evaluated"
                );
            }
            Err(error) => {
                outcome.errors += 1;
                if let Some(api_error) = api_error_details(&error) {
                    outcome.permanent_error |= api_error.is_permanent();
                    outcome.retry_after = outcome.retry_after.max(api_error.retry_after());
                }
                error!(route = %route.name, error = ?error, "route evaluation failed");
            }
        }
    }
    outcome
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}
