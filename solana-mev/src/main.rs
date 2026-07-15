use std::{path::PathBuf, time::Duration};

use anyhow::{anyhow, bail, Result};
use clap::{Parser, ValueEnum};
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

    /// Validate the TOML configuration without requiring credentials.
    #[arg(long)]
    validate_config: bool,

    /// Log format for terminals or machine ingestion.
    #[arg(long, value_enum, default_value_t = LogFormat::Human)]
    log_format: LogFormat,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LogFormat {
    Human,
    Json,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    init_tracing(args.log_format)?;
    let config = Config::load(&args.config)?;
    if args.validate_config {
        println!("configuration is valid: {}", args.config.display());
        return Ok(());
    }
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
        version = env!("CARGO_PKG_VERSION"),
        config_schema = config.schema_version,
        "starting dry-run monitor; transaction execution is intentionally disabled"
    );

    let mut consecutive_failed_scans = 0_u32;
    let mut scan_id = 0_u64;
    loop {
        scan_id = scan_id.wrapping_add(1);
        let outcome = tokio::select! {
            outcome = scan_once(&scanner, &routes, scan_id) => outcome,
            signal = tokio::signal::ctrl_c() => {
                signal?;
                info!(scan_id, "shutdown requested during active scan");
                break;
            }
        };
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
                scan_id,
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
    opportunities: usize,
    permanent_error: bool,
    retry_after: Option<Duration>,
}

async fn scan_once(scanner: &Scanner, routes: &[RouteConfig], scan_id: u64) -> ScanOutcome {
    let started = tokio::time::Instant::now();
    let mut outcome = ScanOutcome::default();
    for route in routes {
        match scanner.evaluate(route).await {
            Ok(evaluation) if evaluation.is_opportunity => {
                outcome.opportunities += 1;
                warn!(
                    scan_id,
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
                    cycle_duration_ms = evaluation.cycle_duration_ms,
                    "candidate opportunity detected; this is not an execution guarantee"
                );
            }
            Ok(evaluation) => {
                info!(
                    scan_id,
                    route = %evaluation.route_name,
                    estimated_profit_bps = evaluation.estimated_profit_bps,
                    minimum_final_amount = evaluation.minimum_final_amount,
                    forward_venues = ?evaluation.forward_venues,
                    return_venues = ?evaluation.return_venues,
                    venues_are_different = evaluation.venues_are_different,
                    cycle_duration_ms = evaluation.cycle_duration_ms,
                    "route evaluated"
                );
            }
            Err(error) => {
                outcome.errors += 1;
                if let Some(api_error) = api_error_details(&error) {
                    outcome.permanent_error |= api_error.is_permanent();
                    outcome.retry_after = outcome.retry_after.max(api_error.retry_after());
                }
                error!(scan_id, route = %route.name, error = ?error, "route evaluation failed");
            }
        }
    }
    info!(
        scan_id,
        routes = routes.len(),
        errors = outcome.errors,
        opportunities = outcome.opportunities,
        elapsed_ms = started.elapsed().as_millis(),
        "scan completed"
    );
    outcome
}

fn init_tracing(format: LogFormat) -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    let result = match format {
        LogFormat::Human => builder.try_init(),
        LogFormat::Json => builder.json().flatten_event(true).try_init(),
    };
    result.map_err(|error| anyhow!("failed to initialize logging: {error}"))?;
    Ok(())
}
