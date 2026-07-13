use std::{path::PathBuf, time::Duration};

use anyhow::Result;
use clap::Parser;
use mev_bot_solana::{
    config::{Config, RouteConfig},
    jupiter::JupiterClient,
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

    loop {
        scan_once(&scanner, &routes).await;
        if args.once {
            break;
        }

        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(config.scanner.interval_ms)) => {}
            signal = tokio::signal::ctrl_c() => {
                signal?;
                info!("shutdown requested");
                break;
            }
        }
    }

    Ok(())
}

async fn scan_once(scanner: &Scanner, routes: &[RouteConfig]) {
    for route in routes {
        match scanner.evaluate(route).await {
            Ok(evaluation) if evaluation.is_opportunity => {
                warn!(
                    route = %evaluation.route_name,
                    start_amount = evaluation.start_amount,
                    intermediate_amount = evaluation.intermediate_amount,
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
                error!(route = %route.name, error = %error, "route evaluation failed");
            }
        }
    }
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}
