mod dex;
mod price_fetcher;
mod api;
mod proto;

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use std::time::Duration;
use tokio::time;
use solana_client::rpc_client::RpcClient;
use std::sync::Arc;
use dotenv::dotenv;
use std::time::Instant;
use crate::price_fetcher::PriceFetcher;
use crate::dex::raydium::RaydiumDex;
use crate::proto::InstantNodeClient;
use tonic::transport::Channel;
use crate::api::start_api_server;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    
    println!("Starting multi-DEX price fetcher...");
    
    let rpc_url = std::env::var("RPC_URL").context("RPC_URL not set")?;
    let rpc_client = Arc::new(RpcClient::new(rpc_url));
    
    // Initialize legacy price fetcher for RPC-based fetching
    let mut legacy_price_fetcher = PriceFetcher::new(rpc_client.clone());
    
    // Add Raydium DEX
    let raydium_program_id = std::env::var("RAYDIUM_PROGRAM_ID")
        .context("RAYDIUM_PROGRAM_ID not set")?;
    legacy_price_fetcher.add_dex(Box::new(RaydiumDex::new(&raydium_program_id)?));

    let update_interval = std::env::var("UPDATE_INTERVAL")
        .unwrap_or_else(|_| "300".to_string())
        .parse::<u64>()
        .context("Failed to parse UPDATE_INTERVAL")?;

    let data_dir = Path::new("data");
    fs::create_dir_all(data_dir).context("Failed to create data directory")?;

    // Create gRPC channel for InstantNode (if enabled)
    let use_instant_node = std::env::var("USE_INSTANT_NODE")
        .unwrap_or_else(|_| "false".to_string())
        .parse::<bool>()
        .unwrap_or(false);

    let mut price_fetcher = legacy_price_fetcher.clone();

    if use_instant_node {
        println!("Using InstantNode for price fetching...");
        
        let endpoint = std::env::var("INSTANT_NODE_ENDPOINT")
            .unwrap_or_else(|_| "https://solana-grpc-geyser.instantnodes.io:443".to_string());
        
        let token = std::env::var("INSTANT_NODE_TOKEN").ok();
        
        // Create gRPC channel
        let channel = Channel::from_shared(endpoint.clone())?
            .connect()
            .await?;
        
        // Create InstantNode client
        let instant_node = InstantNodeClient::new(
            channel,
            endpoint,
            token,
        );
        
        // Create new price fetcher with InstantNode
        price_fetcher = PriceFetcher::new_with_instant_node(instant_node);
        
        // Start price monitoring
        price_fetcher.start().await?;
        
        // Start API server
        let api_port = std::env::var("API_PORT")
            .unwrap_or_else(|_| "3000".to_string())
            .parse::<u16>()
            .unwrap_or(3000);
        
        // Start API server in a separate task
        let price_fetcher_clone = price_fetcher.clone();
        tokio::spawn(async move {
            start_api_server(Arc::new(price_fetcher_clone), api_port).await;
        });
    }

    // Use legacy RPC polling method
    let mut interval = time::interval(Duration::from_secs(update_interval));
    
    loop {
        let start = Instant::now();
        interval.tick().await;
        
        if !use_instant_node {
            match legacy_price_fetcher.fetch_all_prices().await {
                Ok(prices) => {
                    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
                    let price_file = data_dir.join(format!("token_prices_{}.log", timestamp));
                    
                    if let Err(e) = PriceFetcher::save_price_data(&prices, &price_file).await {
                        eprintln!("Error saving price data: {}", e);
                    }

                    let duration = start.elapsed();
                    println!("Fetched {} prices in {:?}", prices.len(), duration);
                }
                Err(e) => {
                    eprintln!("Error fetching prices: {}", e);
                }
            }
        } else {
            // When using InstantNode, just log the current cache stats
            let prices = price_fetcher.get_all_prices();
            println!("Current price cache has {} tokens", prices.len());
            
            // Sleep to not spam logs
            tokio::time::sleep(Duration::from_secs(10)).await;
        }
    }
}
