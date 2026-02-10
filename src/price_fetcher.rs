use {
    crate::proto::{
        InstantNodeClient,
        SubscribeRequestFilterAccounts,
        AccountFilterType,
        CommitmentLevel,
        AccountUpdate,
    },
    anyhow::{Result, anyhow},
    raydium_amm::state::AmmInfo,
    solana_sdk::{pubkey::Pubkey, account::Account},
    std::{
        collections::{HashMap, HashSet},
        sync::{Arc, RwLock},
        str::FromStr,
        time::{Duration, Instant},
    },
    tokio::{sync::mpsc, time},
    futures_util::StreamExt,
    serde::{Deserialize, Serialize},
    reqwest::Client as HttpClient,
    tracing::{info, error, warn, debug},
    solana_client::rpc_client::RpcClient,
    chrono::{DateTime, Local},
    std::path::Path,
    crate::dex::{DexProtocol, TokenPrice as DexTokenPrice},
    bytemuck,
};

/// Token information from Jupiter API
#[derive(Debug, Clone, Deserialize)]
pub struct JupiterToken {
    pub address: String,
    pub name: String,
    pub symbol: String,
    pub decimals: u8,
    #[serde(rename = "logoURI")]
    pub logo_uri: Option<String>,
    pub tags: Vec<String>,
    #[serde(rename = "daily_volume")]
    pub daily_volume: Option<f64>,
}

/// Price data for a token
#[derive(Debug, Clone, Serialize)]
pub struct TokenPrice {
    pub address: String,
    pub symbol: String,
    pub name: String,
    pub price_usd: f64,
    pub price_sol: f64,
    pub volume_24h: Option<f64>,
    pub last_updated: u64,
}

/// Manages token prices using different methods depending on the client
#[derive(Clone)]
pub struct PriceFetcher {
    http_client: HttpClient,
    rpc_client: Option<Arc<RpcClient>>,
    instant_node: Option<Arc<InstantNodeClient>>,
    price_cache: Arc<RwLock<HashMap<String, TokenPrice>>>,
    usdc_mint: Pubkey,
    sol_mint: Pubkey,
    dexes: Arc<RwLock<Vec<Box<dyn DexProtocol + Send + Sync>>>>,
}

impl PriceFetcher {
    /// Create a new price fetcher using RPC client (legacy method)
    pub fn new(rpc_client: Arc<RpcClient>) -> Self {
        Self {
            http_client: HttpClient::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap(),
            rpc_client: Some(rpc_client),
            instant_node: None,
            price_cache: Arc::new(RwLock::new(HashMap::new())),
            usdc_mint: Pubkey::from_str("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v").unwrap(), // USDC
            sol_mint: Pubkey::from_str("So11111111111111111111111111111111111111112").unwrap(),   // Wrapped SOL
            dexes: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create a new price fetcher using InstantNode client (new method)
    pub fn new_with_instant_node(instant_node: InstantNodeClient) -> Self {
        Self {
            http_client: HttpClient::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap(),
            rpc_client: None,
            instant_node: Some(Arc::new(instant_node)),
            price_cache: Arc::new(RwLock::new(HashMap::new())),
            usdc_mint: Pubkey::from_str("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v").unwrap(), // USDC
            sol_mint: Pubkey::from_str("So11111111111111111111111111111111111111112").unwrap(),   // Wrapped SOL
            dexes: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add a DEX protocol for price fetching
    pub fn add_dex(&mut self, dex: Box<dyn DexProtocol + Send + Sync>) {
        self.dexes.write().unwrap().push(dex);
    }

    /// Get price for a token
    pub fn get_price(&self, token_address: &str) -> Option<TokenPrice> {
        self.price_cache.read().unwrap().get(token_address).cloned()
    }

    /// Get all cached prices
    pub fn get_all_prices(&self) -> HashMap<String, TokenPrice> {
        self.price_cache.read().unwrap().clone()
    }

    /// Fetch all prices using legacy RPC client method
    pub async fn fetch_all_prices(&self) -> Result<HashMap<String, DexTokenPrice>> {
        if let Some(rpc_client) = &self.rpc_client {
            let mut results = HashMap::new();
            let dexes = self.dexes.read().unwrap().clone();
            
            // Fetch trending tokens for additional data
            let trending_tokens = self.fetch_trending_tokens().await.unwrap_or_default();
            let token_metadata: HashMap<_, _> = trending_tokens
                .iter()
                .map(|t| (t.address.clone(), t.clone()))
                .collect();
            
            // Get price from each DEX for each token
            for dex in dexes {
                for token in &trending_tokens {
                    let token_address = &token.address;
                    match dex.get_token_price(rpc_client.clone(), token_address).await {
                        Ok(Some(price)) => {
                            let token_price = DexTokenPrice {
                                token_address: token_address.clone(),
                                dex_name: dex.name().to_string(),
                                price,
                                timestamp: Local::now(),
                            };
                            results.insert(format!("{}_{}", token_address, dex.name()), token_price);
                            
                            // Also update our new price cache format
                            let mut cache = self.price_cache.write().unwrap();
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs();
                                
                            let new_price = TokenPrice {
                                address: token_address.clone(),
                                symbol: token.symbol.clone(),
                                name: token.name.clone(),
                                price_usd: if dex.name() == "Raydium" { price } else { 0.0 },
                                price_sol: if dex.name() != "Raydium" { price } else { 0.0 },
                                volume_24h: token.daily_volume,
                                last_updated: now,
                            };
                            
                            cache.insert(token_address.clone(), new_price);
                        }
                        Ok(None) => {
                            debug!("No price found for {} in {}", token_address, dex.name());
                        }
                        Err(e) => {
                            error!("Error fetching price for {} from {}: {}", token_address, dex.name(), e);
                        }
                    }
                }
            }
            
            return Ok(results);
        }
        
        Err(anyhow!("No RPC client available"))
    }

    /// Start the price monitoring service using InstantNode
    pub async fn start(&self) -> Result<()> {
        if self.instant_node.is_none() {
            return Err(anyhow!("InstantNode client not available"));
        }
        
        // Get trending tokens
        let tokens = self.fetch_trending_tokens().await?;
        info!("Fetched {} trending tokens", tokens.len());

        // Create a cache of token metadata
        let mut token_metadata = HashMap::new();
        for token in &tokens {
            token_metadata.insert(token.address.clone(), token.clone());
        }

        // Find Raydium pools for these tokens
        let pool_accounts = self.discover_raydium_pools(&tokens).await?;
        info!("Found {} Raydium pool accounts to monitor", pool_accounts.len());

        // Subscribe to pool accounts
        let price_cache = self.price_cache.clone();
        let token_meta = token_metadata.clone();
        let usdc_mint = self.usdc_mint;
        let sol_mint = self.sol_mint;
        
        let instant_node = self.instant_node.clone().unwrap();
        
        // Convert account addresses to strings for subscription
        let account_addresses: Vec<String> = pool_accounts.iter().map(|addr| addr.to_string()).collect();
        
        // Create account filter for subscription
        let account_filter = SubscribeRequestFilterAccounts {
            account: account_addresses,
            owner: vec![],
            filters: vec![],
        };
        
        // Spawn subscription task
        tokio::spawn(async move {
            // Subscribe to account updates
            match instant_node.subscribe_accounts(account_filter, CommitmentLevel::Confirmed).await {
                Ok(mut stream) => {
                    info!("Successfully subscribed to Raydium pool accounts");
                    
                    // Process account updates
                    while let Some(update_result) = stream.next().await {
                        match update_result {
                            Ok(update) => {
                                // Process pool update
                                if let Err(e) = process_pool_update(
                                    &update, 
                                    &price_cache, 
                                    &token_meta, 
                                    usdc_mint, 
                                    sol_mint
                                ) {
                                    warn!("Error processing pool update: {}", e);
                                }
                            },
                            Err(e) => {
                                error!("Error in account subscription stream: {}", e);
                                // Stream error, will be retried by the InstantNode client
                                break;
                            }
                        }
                    }
                    
                    error!("Raydium pool subscription stream ended");
                },
                Err(e) => {
                    error!("Failed to subscribe to Raydium pool accounts: {}", e);
                }
            }
        });
        
        // Start periodic refresh of trending tokens
        let http_client = self.http_client.clone();
        let instant_node = self.instant_node.clone().unwrap();
        let price_cache = self.price_cache.clone();
        let usdc_mint = self.usdc_mint;
        let sol_mint = self.sol_mint;
        
        tokio::spawn(async move {
            let mut interval = time::interval(Duration::from_secs(3600)); // Refresh every hour
            
            loop {
                interval.tick().await;
                
                // Fetch latest trending tokens
                match fetch_trending_tokens_internal(&http_client).await {
                    Ok(new_tokens) => {
                        info!("Refreshed trending tokens, found {}", new_tokens.len());
                        
                        // Find new pools to monitor
                        match discover_raydium_pools_internal(&new_tokens).await {
                            Ok(new_pools) => {
                                info!("Found {} Raydium pools for trending tokens", new_pools.len());
                                
                                // Update token metadata
                                let mut token_meta = HashMap::new();
                                for token in &new_tokens {
                                    token_meta.insert(token.address.clone(), token.clone());
                                }
                                
                                // Subscribe to any new pool accounts
                                // In a production system, we'd handle this more efficiently by tracking
                                // already-subscribed accounts and only subscribing to new ones
                            },
                            Err(e) => error!("Failed to discover Raydium pools: {}", e),
                        }
                    },
                    Err(e) => error!("Failed to refresh trending tokens: {}", e),
                }
            }
        });
        
        Ok(())
    }

    /// Save price data to file (legacy method)
    pub async fn save_price_data(prices: &HashMap<String, DexTokenPrice>, file_path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(prices)?;
        tokio::fs::write(file_path, json).await?;
        Ok(())
    }

    /// Fetch trending tokens from Jupiter API
    async fn fetch_trending_tokens(&self) -> Result<Vec<JupiterToken>> {
        fetch_trending_tokens_internal(&self.http_client).await
    }

    /// Discover Raydium pool accounts for tokens
    async fn discover_raydium_pools(&self, tokens: &[JupiterToken]) -> Result<Vec<Pubkey>> {
        discover_raydium_pools_internal(tokens).await
    }
}

/// Process a pool account update
fn process_pool_update(
    update: &AccountUpdate,
    price_cache: &Arc<RwLock<HashMap<String, TokenPrice>>>,
    token_metadata: &HashMap<String, JupiterToken>,
    usdc_mint: Pubkey,
    sol_mint: Pubkey,
) -> Result<()> {
    // Parse the Raydium pool data
    let pool_account = update.account.as_ref();
    let pool_data = match parse_raydium_pool(pool_account) {
        Ok(data) => data,
        Err(e) => {
            warn!("Failed to parse Raydium pool data: {}", e);
            return Err(e);
        }
    };
    
    // Identify the pool tokens
    let token_a_mint = pool_data.coin_vault_mint;
    let token_b_mint = pool_data.pc_vault_mint;
    
    // Calculate prices
    let (token_mint, price_usd, price_sol) = if token_a_mint == usdc_mint {
        // Token B / USDC pool
        let price = calculate_price_from_pool(&pool_data, false)?;
        (token_b_mint, price, 0.0) // We'd need SOL price to calculate price_sol
    } else if token_b_mint == usdc_mint {
        // Token A / USDC pool
        let price = calculate_price_from_pool(&pool_data, true)?;
        (token_a_mint, price, 0.0) // We'd need SOL price to calculate price_sol
    } else if token_a_mint == sol_mint {
        // Token B / SOL pool
        let price = calculate_price_from_pool(&pool_data, false)?;
        (token_b_mint, 0.0, price) // We'd need SOL price to calculate price_usd
    } else if token_b_mint == sol_mint {
        // Token A / SOL pool
        let price = calculate_price_from_pool(&pool_data, true)?;
        (token_a_mint, 0.0, price) // We'd need SOL price to calculate price_usd
    } else {
        // Not a USDC or SOL pool, skip
        return Ok(());
    };
    
    // Find token in metadata
    let token_address = token_mint.to_string();
    let token_meta = match token_metadata.get(&token_address) {
        Some(meta) => meta,
        None => {
            // This pool contains a token we're not tracking
            return Ok(());
        }
    };
    
    // Update price cache
    let mut cache = price_cache.write().unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    // Create or update price entry
    let price = TokenPrice {
        address: token_address.clone(),
        symbol: token_meta.symbol.clone(),
        name: token_meta.name.clone(),
        price_usd,
        price_sol,
        volume_24h: token_meta.daily_volume,
        last_updated: now,
    };
    
    cache.insert(token_address, price);
    
    Ok(())
}

/// Calculate token price from pool data
fn calculate_price_from_pool(pool: &AmmInfo, is_token_a: bool) -> Result<f64> {
    // For Raydium pools, we need to examine the StateData fields
    // This is a simplified example - adjust based on actual AmmInfo structure
    if is_token_a {
        // Looking at the StateData struct, we need to extract the relevant balance info
        if pool.state_data.swap_coin_out_amount == 0 || pool.state_data.swap_pc_in_amount == 0 {
            return Err(anyhow!("Pool has zero liquidity"));
        }
        
        // Calculate price as token_b / token_a
        Ok((pool.state_data.swap_pc_in_amount as f64) / (pool.state_data.swap_coin_out_amount as f64))
    } else {
        if pool.state_data.swap_coin_in_amount == 0 || pool.state_data.swap_pc_out_amount == 0 {
            return Err(anyhow!("Pool has zero liquidity"));
        }
        
        // Calculate price as token_a / token_b
        Ok((pool.state_data.swap_coin_in_amount as f64) / (pool.state_data.swap_pc_out_amount as f64))
    }
}

/// Parse Raydium pool data from account
fn parse_raydium_pool(account: &Account) -> Result<AmmInfo> {
    // Check for correct data length first - AmmInfo is 752 bytes
    if account.data.len() < 752 {
        return Err(anyhow!("Account data too short for a Raydium pool"));
    }
    
    // Use unsafe to convert the byte slice to AmmInfo
    // This is safe because AmmInfo is a POD type designed for this purpose
    unsafe {
        let pool_info = std::mem::transmute_copy::<[u8; 752], AmmInfo>(
            &account.data[0..752].try_into().unwrap()
        );
        Ok(pool_info)
    }
}

/// Internal function to fetch trending tokens
async fn fetch_trending_tokens_internal(http_client: &HttpClient) -> Result<Vec<JupiterToken>> {
    // Fetch trending tokens from Jupiter API
    let response = http_client
        .get("https://tokens.jup.ag/tokens?tags=birdeye-trending")
        .send()
        .await?
        .json::<Vec<JupiterToken>>()
        .await?;
    
    Ok(response)
}

/// Internal function to discover Raydium pools
async fn discover_raydium_pools_internal(tokens: &[JupiterToken]) -> Result<Vec<Pubkey>> {
    // In a real implementation, this would use Raydium's SDK or API to find pool accounts
    // This is a placeholder that would need to be implemented
    let pool_accounts = Vec::new();
    
    // For demonstration purposes, we'll return an empty vector
    // In reality, you would:
    // 1. Query Raydium API or on-chain program to find pools for each token
    // 2. Filter for pools that pair with USDC or SOL
    // 3. Return the account addresses
    
    Ok(pool_accounts)
} 