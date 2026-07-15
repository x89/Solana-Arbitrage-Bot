use {
    anchor_client::{
        solana_sdk::{
            commitment_config::CommitmentConfig,
            pubkey::Pubkey,
            signature::{Keypair, Signer},
            system_program,
        },
        Client, Program,
    },
    arbitrage::program::Arbitrage,
    std::{rc::Rc, str::FromStr, sync::Arc},
    tokio::{sync::RwLock, time::{sleep, Duration}},
};

mod price_monitor;
mod route_finder;
mod dex_client;
mod types;

use price_monitor::PriceMonitor;
use route_finder::RouteFinder;
use dex_client::{DexClient, MeteoraClient, RaydiumClient, OrcaClient};
use types::{Route, SwapStep, PriceData, ArbitrageOpportunity};

const MIN_PROFIT_THRESHOLD: f64 = 0.005; // 0.5%
const REFRESH_INTERVAL: u64 = 1000; // 1 second
const MAX_SLIPPAGE: f64 = 0.01; // 1%

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize connection to Solana
    let payer = Keypair::from_bytes(&include_bytes!("keypair.json")[..])?;
    let url = "https://api.mainnet-beta.solana.com".to_string();
    
    let client = Client::new_with_options(
        cluster,
        Rc::new(payer),
        CommitmentConfig::confirmed(),
    );

    // Initialize DEX clients
    let meteora = Arc::new(MeteoraClient::new(&client));
    let raydium = Arc::new(RaydiumClient::new(&client));
    let orca = Arc::new(OrcaClient::new(&client));

    // Initialize price monitor
    let price_monitor = PriceMonitor::new(vec![
        meteora.clone(),
        raydium.clone(),
        orca.clone(),
    ]);

    // Initialize route finder
    let route_finder = RouteFinder::new(
        MIN_PROFIT_THRESHOLD,
        MAX_SLIPPAGE,
    );

    println!("Starting arbitrage bot...");
    
    loop {
        // 1. Get latest prices
        let prices = price_monitor.get_latest_prices().await?;
        
        // 2. Find profitable routes
        let opportunities = route_finder.find_opportunities(&prices).await?;
        
        // 3. Execute profitable trades
        for opp in opportunities {
            match opp.route_type {
                RouteType::TwoHop => {
                    execute_two_hop_arbitrage(&client, &opp).await?;
                },
                RouteType::Triangle => {
                    execute_triangle_arbitrage(&client, &opp).await?;
                },
                RouteType::OrcaWhirlpool => {
                    execute_orca_whirlpool_arbitrage(&client, &opp).await?;
                }
            }
        }

        sleep(Duration::from_millis(REFRESH_INTERVAL)).await;
    }
}

async fn execute_two_hop_arbitrage(
    client: &Program<Arbitrage>,
    opportunity: &ArbitrageOpportunity,
) -> Result<(), Box<dyn std::error::Error>> {
    let accounts = get_two_hop_accounts(client, opportunity)?;
    
    client.request()
        .accounts(accounts)
        .args(instruction::ExecuteTwoHopArbitrage {
            amount_in: opportunity.amount_in,
            minimum_intermediate_amount: opportunity.intermediate_amounts[0],
            minimum_final_amount: opportunity.expected_output,
        })
        .send()?;
    
    println!("Executed two-hop arbitrage: {:?}", opportunity);
    Ok(())
}

async fn execute_triangle_arbitrage(
    client: &Program<Arbitrage>,
    opportunity: &ArbitrageOpportunity,
) -> Result<(), Box<dyn std::error::Error>> {
    let accounts = get_triangle_accounts(client, opportunity)?;
    
    client.request()
        .accounts(accounts)
        .args(instruction::ExecuteTriangleArbitrage {
            amount_in: opportunity.amount_in,
            minimum_amount_1: opportunity.intermediate_amounts[0],
            minimum_amount_2: opportunity.intermediate_amounts[1],
            minimum_final_amount: opportunity.expected_output,
        })
        .send()?;
    
    println!("Executed triangle arbitrage: {:?}", opportunity);
    Ok(())
}

async fn execute_orca_whirlpool_arbitrage(
    client: &Program<Arbitrage>,
    opportunity: &ArbitrageOpportunity,
) -> Result<(), Box<dyn std::error::Error>> {
    let accounts = get_orca_whirlpool_accounts(client, opportunity)?;
    
    client.request()
        .accounts(accounts)
        .args(instruction::ExecuteOrcaWhirlpoolArbitrage {
            amount_in: opportunity.amount_in,
            minimum_amount_1: opportunity.intermediate_amounts[0],
            minimum_final_amount: opportunity.expected_output,
        })
        .send()?;
    
    println!("Executed Orca-Whirlpool arbitrage: {:?}", opportunity);
    Ok(())
} 