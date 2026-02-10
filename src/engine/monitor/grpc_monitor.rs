use {
    crate::{
        common::{logger::Logger, utils::AppState},
        dex::pump_fun::{Pump, execute_swap, PUMP_PROGRAM_ID, PUMP_BUY_METHOD, PUMP_SELL_METHOD},
    },
    anyhow::{anyhow, Context, Result},
    backoff::{future::retry, ExponentialBackoff},
    std::time::Duration,
    yellowstone_grpc_client::GeyserGrpcClient,
    yellowstone_grpc_proto::{
        prelude::{
            subscribe_update::UpdateOneof,
            CommitmentLevel,
            SubscribeRequest,
            SubscribeRequestFilterTransactions,
            SubscribeRequestPing,
        },
        convert_from,
        geyser::geyser_client::GeyserClient,
    },
    base64::Engine as _,
    bs58,
    chrono::Utc,
    solana_sdk::{signature::Signer, pubkey::Pubkey},
    solana_transaction_status::UiTransactionEncoding,
    tonic::{
        transport::{Channel, ClientTlsConfig},
        metadata::MetadataValue,
        Request,
    },
    tonic_health::pb::health_client::HealthClient,
    futures_util::{SinkExt, StreamExt},
    std::{collections::HashMap, str::FromStr, sync::Arc},
    tokio::time::sleep,
};
use crate::dex::pump_fun::PumpInfo;

const TARGET_WALLET: &str = "o7RY6P2vQMuGSu1TrLM81weuzgDjaCRTXYRaXJwWcvc";
const RETRY_DELAY: u64 = 5; // seconds
const MAX_PUMP_INFO_RETRIES: u8 = 3;
const RETRY_DELAY_MS: u64 = 100;

pub async fn monitor_transactions_grpc(
    grpc_url: &str,
    state: AppState,
) -> Result<()> {
    let logger = Logger::new("[GRPC-MONITOR]".to_string());

    let backoff = ExponentialBackoff {
        initial_interval: Duration::from_secs(1),
        max_interval: Duration::from_secs(60),
        multiplier: 1.5,
        max_elapsed_time: None,
        ..Default::default()
    };

    retry(backoff, || {
        let state = state.clone();
        let grpc_url = grpc_url.to_string();
        let logger = logger.clone();

        async move {
            logger.info("Attempting to connect to gRPC server...".to_string());

            // Create gRPC client with TLS and token
            let channel = Channel::from_shared(grpc_url.clone())
                .map_err(|e| backoff::Error::permanent(anyhow!("Invalid URI: {}", e)))?
                .tls_config(ClientTlsConfig::new())
                .map_err(|e| backoff::Error::permanent(anyhow!("TLS config error: {}", e)))?
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(10))
                .connect()
                .await
                .map_err(|e| backoff::Error::transient(anyhow!("Failed to create channel: {}", e)))?;

            let token = std::env::var("RPC_TOKEN")
                .map_err(|e| backoff::Error::permanent(anyhow!("Failed to get RPC_TOKEN: {}", e)))?;
            let token_value = MetadataValue::try_from(token)
                .map_err(|e| backoff::Error::permanent(anyhow!("Invalid token format: {}", e)))?;

            let interceptor = move |mut req: Request<()>| {
                req.metadata_mut().insert("x-token", token_value.clone());
                Ok(req)
            };

            let health_client = HealthClient::with_interceptor(channel.clone(), interceptor.clone());
            let geyser_client = GeyserClient::with_interceptor(channel, interceptor);
            let mut client = GeyserGrpcClient::new(health_client, geyser_client);

            logger.success("Successfully connected to gRPC server".to_string());

            // Create subscription request
            let request = create_subscription_request()
                .map_err(|e| backoff::Error::permanent(anyhow!("Failed to create request: {}", e)))?;

            // Subscribe and handle stream
            let (mut subscribe_tx, mut stream) = client.subscribe().await
                .map_err(|e| backoff::Error::transient(anyhow!("Failed to create subscription: {}", e)))?;

            subscribe_tx.send(request).await
                .map_err(|e| backoff::Error::transient(anyhow!("Failed to send request: {}", e)))?;

            logger.info("Started monitoring PumpFun transactions...".to_string());

            while let Some(message) = stream.next().await {
                match message {
                    Ok(msg) => {
                        if let Some(UpdateOneof::Transaction(tx)) = msg.update_oneof {
                            if let Err(e) = process_transaction(&logger, &state, tx).await {
                                logger.error(format!("Failed to process transaction: {}", e));
                            }
                        } else if let Some(UpdateOneof::Ping(_)) = msg.update_oneof {
                            if let Err(e) = subscribe_tx.send(SubscribeRequest {
                                ping: Some(SubscribeRequestPing { id: 1 }),
                                ..Default::default()
                            }).await {
                                logger.error(format!("Failed to send ping: {}", e));
                            }
                        }
                    }
                    Err(e) => {
                        logger.error(format!("Stream error: {}", e));
                        return Err(backoff::Error::transient(anyhow!(e)));
                    }
                }
            }

            logger.warning("Stream closed, attempting to reconnect...".to_string());
            Err(backoff::Error::transient(anyhow!("Stream closed")))
        }
    })
    .await?;

    Ok(())
}

fn create_subscription_request() -> Result<SubscribeRequest> {
    let transaction_filter = SubscribeRequestFilterTransactions {
        vote: Some(false),
        failed: Some(false),
        signature: None,
        account_include: vec![TARGET_WALLET.to_string()],
        account_exclude: vec![],
        account_required: vec![],
    };

    let mut transaction_filters = HashMap::new();
    transaction_filters.insert("pump_fun".to_string(), transaction_filter);

    Ok(SubscribeRequest {
        transactions: transaction_filters,
        accounts: Default::default(),
        accounts_data_slice: Default::default(),
        blocks: Default::default(),
        blocks_meta: Default::default(),
        commitment: Some(CommitmentLevel::Confirmed as i32),
        entry: Default::default(),
        ping: Default::default(),
        slots: Default::default(),
        transactions_status: Default::default(),
    })
}

async fn process_transaction(
    logger: &Logger,
    state: &AppState,
    tx: yellowstone_grpc_proto::prelude::SubscribeUpdateTransaction,
) -> Result<()> {
    let start_time = std::time::Instant::now();

    if let Some(tx_data) = &tx.transaction {
        if let Some(meta) = &tx_data.meta {
            let logs = &meta.log_messages;
            
            // Log transaction details - convert signature bytes to base58 string
            let signature = bs58::encode(&tx_data.signature).into_string();

            logger.info(format!(
                "\n   * [NEW TX] => (\"{}\") - SLOT:({})\n   * [FROM] => ({})\n   * [TIME] => {} :: ({:?}).",
                signature,
                tx_data.index,
                TARGET_WALLET,
                Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true),
                start_time.elapsed()
            ));

            // Log full transaction logs for debugging
            logger.debug(format!("\n   * [TRANSACTION LOGS] => {:#?}", logs));
            
            // Only process if it's a PumpFun transaction
            if logs.iter().any(|log| log.contains(PUMP_PROGRAM_ID)) {
                logger.success("Found PumpFun transaction!".to_string());
                logger.info(format!(
                    "\n   * [PUMP TRANSACTION FOUND] =>\n   * [LOGS] => {:#?}",
                    logs
                ));
                
                // Extract mint address and instruction type from logs
                let mut mint_address = String::new();
                let mut is_buy = false;
                let mut transaction_amount: Option<u64> = None;
                
                for log in logs {
                    // Check for instruction type in program logs
                    if log.starts_with("Program log: Instruction:") {
                        let instruction = log.trim_start_matches("Program log: Instruction:");
                        match instruction.trim() {
                            "Buy" => {
                                logger.success("Identified as BUY instruction".to_string());
                                is_buy = true;
                            },
                            "Sell" => {
                                logger.success("Identified as SELL instruction".to_string());
                                is_buy = false;
                            },
                            _ => {
                                logger.debug(format!("Skipping non-buy/sell instruction: {}", instruction));
                                continue;
                            }
                        }
                    }
                    
                    // Extract program data for mint address and amount
                    if log.starts_with("Program data: ") {
                        let data = log.trim_start_matches("Program data: ");
                        if let Ok(decoded) = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data) {
                            logger.success(format!("Decoded data length: {}", decoded.len()));
                            
                            // Next 32 bytes after instruction data should be the mint address
                            if decoded.len() >= 40 {  // 8 (discriminator) + 32 (mint)
                                let mint_bytes = &decoded[8..40];
                                mint_address = bs58::encode(mint_bytes).into_string();
                                logger.success(format!("Extracted mint address: {}", mint_address));
                            }

                            // Log amount if available
                            if decoded.len() >= 48 {
                                let amount_bytes = &decoded[40..48];
                                let amount = u64::from_le_bytes(amount_bytes.try_into().unwrap());
                                transaction_amount = Some(amount);
                                logger.success(format!("Detected transaction amount: {} SOL", 
                                    amount as f64 / 1_000_000_000.0));
                            }
                        } else {
                            logger.error("Failed to decode base64 data".to_string());
                        }
                    }
                }

                logger.debug(format!(
                    "Final extracted info - Mint: {}, Is Buy: {}",
                    mint_address, is_buy
                ));

                if mint_address.is_empty() {
                    return Err(anyhow!("Could not extract mint address from logs"));
                }

                // Validate the extracted mint address
                if let Err(_) = Pubkey::from_str(&mint_address) {
                    return Err(anyhow!("Invalid mint address format"));
                }

                logger.info(format!(
                    "\n   * [BUILD-IXN]({}) - {} :: {:?}",
                    mint_address,
                    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true),
                    start_time.elapsed()
                ));

                // Create Pump instance
                let pump = Pump::new(
                    state.rpc_nonblocking_client.clone(),
                    state.wallet.clone(),
                );

                // Execute swap with retry logic
                let swap_start = std::time::Instant::now();
                match execute_swap(&pump, &mint_address, is_buy, state.scale, transaction_amount).await {
                    Ok(signature) => {
                        let swap_time = swap_start.elapsed();
                        let total_time = start_time.elapsed();
                        logger.success(format!(
                            "\n   * [SUCCESSFUL-{}] => TX_HASH: (\"{}\") \n   * [POOL] => ({}) \n   * [TIMING] => Swap: {}ms, Total: {}ms \n   * [COPIED] => {}",
                            if is_buy { "BUY" } else { "SELL" },
                            signature,
                            mint_address,
                            swap_time.as_millis(),
                            total_time.as_millis(),
                            Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true)
                        ));
                    }
                    Err(e) => {
                        let total_time = start_time.elapsed();
                        logger.error(format!(
                            "\n   * [FAILED-{}] => Error: {} \n   * [POOL] => ({}) \n   * [TIMING] => Failed after {}ms \n   * [TIME] => {}",
                            if is_buy { "BUY" } else { "SELL" },
                            e,
                            mint_address,
                            total_time.as_millis(),
                            Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true)
                        ));
                    }
                }
            }
        }
    }

    Ok(())
} 