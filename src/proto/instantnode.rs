use {
    tonic::{
        transport::Channel,
        Request,
        Response,
        Status,
        metadata::MetadataValue,
    },
    futures_util::Stream,
    anyhow::{Result, anyhow},
    std::{pin::Pin, collections::HashMap, sync::Arc},
    async_stream::stream,
    std::str::FromStr,
    solana_sdk::pubkey::Pubkey,
    tokio::sync::mpsc,
    futures_util::StreamExt,
    solana_sdk::account::Account,
    std::time::Duration,
};

/// Client for interacting with InstantNode gRPC API
#[derive(Debug, Clone)]
pub struct InstantNodeClient {
    endpoint: String,
    channel: Channel,
    token: Option<String>,
}

/// Represents a request to subscribe to various Solana data
#[derive(Debug)]
pub struct SubscribeRequest {
    pub accounts: HashMap<String, SubscribeRequestFilterAccounts>,
    pub slots: HashMap<String, SubscribeRequestFilterSlots>,
    pub transactions: HashMap<String, SubscribeRequestFilterTransactions>,
    pub commitment: Option<CommitmentLevel>,
}

/// Filter for account subscriptions
#[derive(Debug, Clone)]
pub struct SubscribeRequestFilterAccounts {
    pub account: Vec<String>,
    pub owner: Vec<String>,
    pub filters: Vec<SubscribeRequestFilterAccountsFilter>,
}

/// Filter for slot subscriptions
#[derive(Debug)]
pub struct SubscribeRequestFilterSlots {
    pub filter_by_commitment: Option<bool>,
}

/// Filter for transaction subscriptions
#[derive(Debug)]
pub struct SubscribeRequestFilterTransactions {
    pub vote: Option<bool>,
    pub failed: Option<bool>,
    pub account_include: Vec<String>,
}

/// Account filter types
#[derive(Debug, Clone)]
pub enum AccountFilterType {
    Datasize(u64),
    Memcmp(MemcmpFilter),
    TokenAccountState,
}

/// Memcmp filter for filtering accounts by their data
#[derive(Debug, Clone)]
pub struct MemcmpFilter {
    pub offset: u64,
    pub data: MemcmpFilterData,
}

/// Data formats for memcmp filters
#[derive(Debug, Clone)]
pub enum MemcmpFilterData {
    Bytes(Vec<u8>),
    Base58(String),
    Base64(String),
}

/// Account filter wrapper
#[derive(Debug, Clone)]
pub struct SubscribeRequestFilterAccountsFilter {
    pub filter_type: AccountFilterType,
}

/// Commitment level for subscriptions
#[derive(Debug, Clone, Copy)]
pub enum CommitmentLevel {
    Processed = 0,
    Confirmed = 1,
    Finalized = 2,
}

/// Update for transaction subscription
#[derive(Debug)]
pub struct TransactionUpdate {
    pub signature: String,
    pub slot: u64,
    pub err: Option<String>,
    pub logs: Option<Vec<String>>,
    pub accounts: Vec<String>,
    pub timestamp: i64,
}

/// Update for account subscription
#[derive(Debug)]
pub struct AccountUpdate {
    pub pubkey: Pubkey,
    pub account: Arc<Account>,
    pub slot: u64,
    pub commitment: CommitmentLevel,
}

impl InstantNodeClient {
    /// Create a new InstantNode client
    pub fn new(channel: Channel, endpoint: String, token: Option<String>) -> Self {
        Self { channel, endpoint, token }
    }

    /// Subscribe to transactions
    pub async fn subscribe_transactions(
        &mut self,
        mut request: Request<SubscribeRequest>,
    ) -> Result<Response<Pin<Box<dyn Stream<Item = Result<TransactionUpdate, Status>> + Send + 'static>>>> {
        // Add token to request headers if available
        if let Some(token) = &self.token {
            let token_value = MetadataValue::try_from(token)?;
            request.metadata_mut().insert("x-token", token_value);
        }
        
        // Create stream without capturing self
        let channel = self.channel.clone();
        let endpoint = self.endpoint.clone();
        let token = self.token.clone();
        
        let stream = Box::pin(stream! {
            let client = InstantNodeClient::new(channel, endpoint, token);
            loop {
                match client.get_next_transaction().await {
                    Ok(update) => yield Ok(update),
                    Err(e) => {
                        yield Err(Status::internal(format!("Stream error: {}", e)));
                        break;
                    }
                }
            }
        });

        Ok(Response::new(stream))
    }

    /// Subscribe to account updates
    pub async fn subscribe_accounts(
        &self,
        accounts_filter: SubscribeRequestFilterAccounts,
        commitment: CommitmentLevel,
    ) -> Result<impl Stream<Item = Result<AccountUpdate, Status>> + Send + 'static> {
        // Create channel for receiving account updates
        let (tx, rx) = mpsc::channel(100);
        
        // Clone required fields for async task
        let channel = self.channel.clone();
        let endpoint = self.endpoint.clone();
        let token = self.token.clone();
        
        // Spawn background task to handle subscription
        tokio::spawn(async move {
            let mut retry_count = 0;
            let max_retries = 5;
            let mut backoff_duration = Duration::from_secs(1);
            
            'retry_loop: loop {
                // Create subscription request
                let mut subscription_accounts = HashMap::new();
                subscription_accounts.insert("account_subscription".to_string(), accounts_filter.clone());
                
                let request = SubscribeRequest {
                    accounts: subscription_accounts,
                    slots: HashMap::new(),
                    transactions: HashMap::new(),
                    commitment: Some(commitment),
                };
                
                // Create gRPC client request
                let mut req_builder = Request::new(request);
                
                // Add token if available
                if let Some(token_str) = &token {
                    if let Ok(token_value) = MetadataValue::try_from(token_str.as_str()) {
                        req_builder.metadata_mut().insert("x-token", token_value);
                    }
                }
                
                // Initialize stream
                match process_account_subscription(channel.clone(), req_builder).await {
                    Ok(mut stream) => {
                        // Reset retry counter on successful connection
                        retry_count = 0;
                        backoff_duration = Duration::from_secs(1);
                        
                        // Process stream
                        while let Some(update) = stream.next().await {
                            match update {
                                Ok(account_update) => {
                                    // Forward update to channel
                                    if tx.send(Ok(account_update)).await.is_err() {
                                        // Channel closed, exit
                                        break 'retry_loop;
                                    }
                                },
                                Err(e) => {
                                    // Report error but continue
                                    let _ = tx.send(Err(Status::internal(format!("Stream error: {}", e)))).await;
                                    break; // Break out of stream processing to retry
                                }
                            }
                        }
                        
                        // If we get here, the stream ended - we'll retry
                        println!("Account subscription stream ended, retrying...");
                    },
                    Err(e) => {
                        // Report error to channel
                        let _ = tx.send(Err(Status::internal(format!("Failed to create subscription: {}", e)))).await;
                    }
                }
                
                // Retry with exponential backoff
                retry_count += 1;
                if retry_count > max_retries {
                    // Reset retry count but keep increasing backoff
                    retry_count = max_retries / 2;
                    backoff_duration = std::cmp::min(backoff_duration * 2, Duration::from_secs(60));
                }
                
                // Wait before retrying
                tokio::time::sleep(backoff_duration).await;
            }
        });
        
        // Return receiver as stream
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    async fn get_next_transaction(&self) -> Result<TransactionUpdate> {
        // This would be implemented with actual gRPC calls in a real client
        Err(anyhow!("Not implemented: get_next_transaction"))
    }
}

/// Process an account subscription stream
async fn process_account_subscription(
    channel: Channel,
    request: Request<SubscribeRequest>,
) -> Result<impl Stream<Item = Result<AccountUpdate, Status>> + Send + 'static> {
    // This would be implemented with actual gRPC calls in a real client
    // For now, return a placeholder stream that just yields an error
    
    let stream = Box::pin(stream! {
        // In a real implementation, this would connect to the gRPC server and yield account updates
        yield Err(Status::unimplemented("Account subscription not yet implemented"));
    });
    
    Ok(stream)
}

impl From<SubscribeRequest> for tonic::Request<SubscribeRequest> {
    fn from(req: SubscribeRequest) -> Self {
        tonic::Request::new(req)
    }
}

impl From<AccountFilterType> for SubscribeRequestFilterAccountsFilter {
    fn from(filter_type: AccountFilterType) -> Self {
        Self { filter_type }
    }
} 