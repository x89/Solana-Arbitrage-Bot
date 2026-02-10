use {
    anyhow::{Result, anyhow},
    futures_util::{SinkExt, StreamExt},
    serde_json::{json, Value},
    solana_sdk::pubkey::Pubkey,
    std::{
        collections::HashMap,
        str::FromStr,
        sync::Arc,
    },
    tokio::sync::mpsc,
    tokio_tungstenite::{
        connect_async,
        tungstenite::protocol::Message,
    },
    tracing::{info, error, warn},
    crate::price_fetcher::{PriceFetcher, TokenPrice},
};

pub async fn monitor_accounts_ws(
    endpoint: &str,
    accounts: Vec<String>,
    price_fetcher: Arc<PriceFetcher>,
) -> Result<()> {
    info!("Connecting to WebSocket endpoint: {}", endpoint);
    
    // Connect to WebSocket endpoint
    let (ws_stream, _) = connect_async(endpoint).await
        .map_err(|e| anyhow!("Failed to connect to WebSocket: {}", e))?;
    
    info!("WebSocket connection established");
    
    let (mut write, mut read) = ws_stream.split();
    
    // Create a subscription ID map to track which account each update belongs to
    let mut subscription_map = HashMap::new();
    
    // Subscribe to each account
    for (idx, account) in accounts.iter().enumerate() {
        let sub_id = idx + 1; // Use a simple incrementing ID for each subscription
        
        // Create subscription request
        let subscribe_msg = json!({
            "jsonrpc": "2.0",
            "id": sub_id,
            "method": "accountSubscribe",
            "params": [
                account,
                {
                    "encoding": "base64",
                    "commitment": "confirmed"
                }
            ]
        });
        
        // Send subscription request
        write.send(Message::Text(subscribe_msg.to_string())).await
            .map_err(|e| anyhow!("Failed to send subscription request: {}", e))?;
        
        info!("Subscribed to account: {}", account);
    }
    
    // Process incoming messages
    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Parse the JSON message
                let value: Value = serde_json::from_str(&text)
                    .map_err(|e| anyhow!("Failed to parse WebSocket message: {}", e))?;
                
                // Handle subscription confirmations
                if let Some(id) = value.get("id").and_then(|id| id.as_u64()) {
                    if let Some(result) = value.get("result").and_then(|r| r.as_u64()) {
                        // This is a subscription confirmation
                        let idx = (id as usize) - 1;
                        if idx < accounts.len() {
                            subscription_map.insert(result, accounts[idx].clone());
                            info!("Confirmed subscription {} for account {}", result, accounts[idx]);
                        }
                    }
                }
                
                // Handle account updates
                if let Some(params) = value.get("params") {
                    if let Some(subscription) = params.get("subscription").and_then(|s| s.as_u64()) {
                        if let Some(account) = subscription_map.get(&subscription) {
                            // Process the account update
                            info!("Received update for account: {}", account);
                            
                            // Extract account data and process it
                            // This would need to be implemented based on the expected data format
                            // For Raydium pools, you would need to parse the SwapInfo structure
                        }
                    }
                }
            },
            Ok(Message::Binary(_)) => {
                // Ignore binary messages
            },
            Ok(Message::Ping(_)) => {
                // Respond to ping with pong
                write.send(Message::Pong(vec![])).await
                    .map_err(|e| anyhow!("Failed to send pong: {}", e))?;
            },
            Ok(Message::Pong(_)) => {
                // Ignore pong messages
            },
            Ok(Message::Close(_)) => {
                warn!("WebSocket connection closed by server");
                break;
            },
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
        }
    }
    
    Ok(())
} 