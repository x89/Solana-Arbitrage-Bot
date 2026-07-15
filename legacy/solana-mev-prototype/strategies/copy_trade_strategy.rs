use crate::dex::dex_manager::DexManager;
use crate::error::Result;
use crate::models::copy_trade_opportunity::CopyTradeOpportunity;
use crate::models::market::Market;
use crate::models::order::Order;
use solana_client::rpc_client::RpcClient;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct CopyTradeStrategy {
    pub rpc_client: Arc<RpcClient>,
    pub dex_manager: Arc<Mutex<DexManager>>,
    pub tracked_traders: Vec<Pubkey>,
    pub trade_threshold: f64,
    pub max_trade_amount: f64,
}

impl CopyTradeStrategy {
    pub fn new(
        rpc_client: Arc<RpcClient>,
        dex_manager: Arc<Mutex<DexManager>>,
        tracked_traders: Vec<Pubkey>,
        trade_threshold: f64,
        max_trade_amount: f64,
    ) -> Self {
        CopyTradeStrategy {
            rpc_client,
            dex_manager,
            tracked_traders,
            trade_threshold,
            max_trade_amount,
        }
    }

    pub async fn run(&self) -> Result<()> {
        loop {
            let opportunities = self.find_opportunities().await?;
            
            for opportunity in opportunities {
                self.execute_copy_trade(&opportunity).await?;
            }
            
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }

    async fn find_opportunities(&self) -> Result<Vec<CopyTradeOpportunity>> {
        let mut opportunities = Vec::new();
        
        for trader in &self.tracked_traders {
            let trades = self.get_recent_trades(trader).await?;
            
            for trade in trades {
                if trade.quantity >= self.trade_threshold && trade.quantity <= self.max_trade_amount {
                    let market = self.dex_manager.lock().await.get_market(&trade.market).await?;
                    
                    let opportunity = CopyTradeOpportunity {
                        trader: *trader,
                        market,
                        trade,
                    };
                    opportunities.push(opportunity);
                }
            }
        }
        
        Ok(opportunities)
    }

    async fn get_recent_trades(&self, trader: &Pubkey) -> Result<Vec<Order>> {
        let signature_infos = self.rpc_client.get_signatures_for_address(trader)?;
        
        let mut trades = Vec::new();
        
        for signature_info in signature_infos {
            if let Some(signature) = signature_info.signature {
                let transaction = self.rpc_client.get_transaction(&signature)?;
                
                if let Some(transaction) = transaction {
                    for instruction in transaction.transaction.message.instructions {
                        if let Some(dex_instruction) = DexInstruction::unpack(instruction) {
                            match dex_instruction {
                                DexInstruction::NewOrder { .. } => {
                                    let order = self.parse_order(&transaction, &instruction)?;
                                    trades.push(order);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        
        Ok(trades)
    }

    fn parse_order(&self, transaction: &TransactionInfo, instruction: &CompiledInstruction) -> Result<Order> {
        let market_address = instruction.accounts[0];
        let market = self.dex_manager.lock().await.get_market(&market_address).await?;
        
        let side = match instruction.data[0] {
            0 => OrderSide::Bid,
            1 => OrderSide::Ask,
            _ => return Err(anyhow!("Invalid order side")),
        };
        
        let order_type = match instruction.data[1] {
            0 => OrderType::Limit,
            1 => OrderType::ImmediateOrCancel,
            2 => OrderType::PostOnly,
            _ => return Err(anyhow!("Invalid order type")),
        };
        
        let price = f64::from_le_bytes(instruction.data[2..10].try_into()?);
        let quantity = f64::from_le_bytes(instruction.data[10..18].try_into()?);
        
        Ok(Order {
            id: transaction.transaction.signatures[0].to_string(),
            market,
            side,
            order_type,
            price,
            quantity,
            status: OrderStatus::Filled,
        })
    }
    async fn execute_copy_trade(&self, opportunity: &CopyTradeOpportunity) -> Result<()> {
        let market = &opportunity.market;
        let trade = &opportunity.trade;
        let order = self
            .dex_manager
            .lock()
            .await
            .place_order(market, trade.order_type, trade.side, trade
                async fn execute_copy_trade(&self, opportunity: &CopyTradeOpportunity) -> Result<()> {
    let market = &opportunity.market;
    let trade = &opportunity.trade;
    
    let order = self
        .dex_manager
        .lock()
        .await
        .place_order(market, trade.order_type, trade.side, trade.price, trade.quantity)
        .await?;
    
    println!("Placed copy trade order: {:?}", order);
    
    Ok(())
}