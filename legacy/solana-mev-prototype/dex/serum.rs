use crate::dex::dex_trait::DexTrait;
use crate::error::Result;
use crate::models::market::Market;
use crate::models::order::{Order, OrderSide, OrderStatus, OrderType};
use sdk::pubkey::Pubkey;
use sdk::signature::Keypair;
use sdk::signer::Signer;
use sdk::transaction::Transaction;
use solana_client::rpc_client::RpcClient;
use std::collections::HashMap;

pub struct Serum {
    pub rpc_client: RpcClient,
    pub program_id: Pubkey,
    pub authority: Keypair,
}

impl Serum {
    pub fn new(rpc_client: RpcClient, program_id: Pubkey, authority: Keypair) -> Self {
        Serum {
            rpc_client,
            program_id,
            authority,
        }
    }
}

#[async_trait]
impl DexTrait for Serum {
    async fn get_markets(&self) -> Result<Vec<Market>> {
        let mut markets = Vec::new();
        
        let market_infos = self.rpc_client.get_program_accounts(&self.program_id)?;
        
        for market_info in market_infos {
            let market_data: MarketState = bincode::deserialize(&market_info.account.data)?;
            
            let market = Market {
                address: market_info.pubkey,
                name: String::from_utf8_lossy(&market_data.name).to_string(),
                base_asset: market_data.coin_lot_size,
                quote_asset: market_data.pc_lot_size,
                base_decimals: market_data.coin_decimals,
                quote_decimals: market_data.pc_decimals,
            };
            markets.push(market);
        }
        
        Ok(markets)
    }

    async fn get_orderbook(&self, market: &Market) -> Result<(Vec<Order>, Vec<Order>)> {
        let market_account_info = self.rpc_client.get_account_info(&market.address)?;
        let market_data: MarketState = bincode::deserialize(&market_account_info.data)?;
        
        let bids = market_data.load_bids_mut(&self.program_id)?;
        let asks = market_data.load_asks_mut(&self.program_id)?;
        
        let bid_orders = bids.orders(&market_data, &self.program_id)?;
        let ask_orders = asks.orders(&market_data, &self.program_id)?;
        
        Ok((bid_orders, ask_orders))
    }

    async fn place_order(
        &self,
        market: &Market,
        order_type: OrderType,
        side: OrderSide,
        price: f64,
        quantity: f64,
    ) -> Result<Order> {
        let order_id = self.create_order_id();
        let order_account = Keypair::new();
        
        let (token_a_account, token_b_account) = self.get_token_accounts(market).await?;
        
        let order_data = NewOrderInstructionV3 {
            side: side.into(),
            limit_price: price,
            max_qty: quantity,
            order_type: order_type.into(),
            client_order_id: 0,
            self_trade_behavior: SelfTradeBehavior::DecrementTake,
            limit: 65535,
            max_coin_qty: quantity,
            max_native_pc_qty_including_fees: price * quantity,
            self_trade_behavior_v2: SelfTradeBehaviorV2::CancelProvide,
            padding: [0; 5],
        };
        
        let accounts = match side {
            OrderSide::Bid => vec![
                AccountMeta::new(market.address, false),
                AccountMeta::new(order_account.pubkey(), true),
                AccountMeta::new(self.authority.pubkey(), true),
                AccountMeta::new_readonly(spl_token::ID, false),
                AccountMeta::new(token_b_account, false),
                AccountMeta::new(self.get_bids_address(&market)?, false),
                AccountMeta::new(self.get_asks_address(&market)?, false),
                AccountMeta::new(self.get_event_queue_address(&market)?, false),
                AccountMeta::new(token_a_account, false),
                AccountMeta::new_readonly(solana_sdk::sysvar::rent::ID, false),
            ],
            OrderSide::Ask => vec![
                AccountMeta::new(market.address, false),
                AccountMeta::new(order_account.pubkey(), true),
                AccountMeta::new(self.authority.pubkey(), true),
                AccountMeta::new_readonly(spl_token::ID, false),
                AccountMeta::new(token_a_account, false),
                AccountMeta::new(self.get_asks_address(&market)?, false),
                AccountMeta::new(self.get_bids_address(&market)?, false),
                AccountMeta::new(self.get_event_queue_address(&market)?, false),
                AccountMeta::new(token_b_account, false),
                AccountMeta::new_readonly(solana_sdk::sysvar::rent::ID, false),
            ],
        };
        
        let instruction = Instruction {
            program_id: self.program_id,
            accounts,
            data: order_data.pack(),
        };
        
        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        let transaction = Transaction::new_signed_with_payer(
            &[instruction],
            Some(&self.authority.pubkey()),
            &[&self.authority, &order_account],
            recent_blockhash,
        );
        
        self.rpc_client.send_and_confirm_transaction(&transaction)?;
        
        Ok(Order {
            id: order_id,
            market: market.clone(),
            order_type,
            side,
            price,
            quantity,
            status: OrderStatus::Open,
        })
    }

    async fn cancel_order(&self, order: &Order) -> Result<()> {
        let cancel_data = MarketInstruction::CancelOrderV2 {
            side: order.side.into(),
            order_id: order.id,
        };
        
        let accounts = vec![
            AccountMeta::new(order.market.address, false),
            AccountMeta::new_readonly(self.authority.pubkey(), true),
            AccountMeta::new(self.get_bids_address(&order.market)?, false),
            AccountMeta::new(self.get_asks_address(&order.market)?, false),
            AccountMeta::new(self.get_event_queue_address(&order.market)?, false),
        ];
        
        let instruction = Instruction {
            program_id: self.program_id,
            accounts,
            data: cancel_data.pack(),
        };
        
        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        let transaction = Transaction::new_signed_with_payer(
            &[instruction],
            Some(&self.authority.pubkey()),
            &[&self.authority],
            recent_blockhash,
        );
        
        self.rpc_client.send_and_confirm_transaction(&transaction)?;
        
        Ok(())
    }

    async fn get_balances(&self, market: &Market) -> Result<HashMap<Pubkey, f64>> {
        let vault_signer = self.get_vault_signer_address(&market)?;
        let base_vault = self.get_vault_address(&market, &market.base_asset)?;
        let quote_vault = self.get_vault_address(&market, &market.quote_asset)?;
        
        let base_vault_balance = self.rpc_client.get_token_account_balance(&base_vault)?;
        let quote_vault_balance = self.rpc_client.get_token_account_balance(&quote_vault)?;
        
        let mut balances = HashMap::new();
        balances.insert(market.base_asset, base_vault_balance.amount as f64);
        balances.insert(market.quote_asset, quote_vault_balance.amount as f64);
        
        Ok(balances)
    }
}

impl Serum {
    fn create_order_id(&self) -> u64 {
        rand::random()
    }

    async fn get_token_accounts(&self, market: &Market) -> Result<(Pubkey, Pubkey)> {
        let token_a_account = spl_associated_token_account::get_associated_token_address(
            &self.authority.pubkey(),
            &market.base_asset,
        );
        let token_b_account = spl_associated_token_account::get_associated_token_address(
            &self.authority.pubkey(),
            &market.quote_asset,
        );
        
        Ok((token_a_account, token_b_account))
    }

    fn get_bids_address(&self, market: &Market) -> Result<Pubkey> {
        let (bids_address, _) = Pubkey::find_program_address(
            &[
                &market.address.to_bytes(),
                b"bids",
            ],
            &self.program_id,
        );
        
        Ok(bids_address)
    }

    fn get_asks_address(&self, market: &Market) -> Result<Pubkey> {
        let (asks_address, _) = Pubkey::find_program_address(
            &[
                &market.address.to_bytes(),
                b"asks",
            ],
            &self.program_id,
        );
        
        Ok(asks_address)
    }

    fn get_event_queue_address(&self, market: &Market) -> Result<Pubkey> {
        let (event_queue_address, _) = Pubkey::find_program_address(
            &[
                &market.address.to_bytes(),
                b"event_queue",
            ],
            &self.program_id,
        );
        
        Ok(event_queue_address)
    }

    fn get_vault_signer_address(&self, market: &Market) -> Result<Pubkey> {
        let (vault_signer_address, _) = Pubkey::find_program_address(
            &[
                &market.address.to_bytes(),
                b"vault_signer",
            ],
            &self.program_id,
        );
        
        Ok(vault_signer_address)
    }

    fn get_vault_address(&self, market: &Market, token_mint: &Pubkey) -> Result<Pubkey> {
        let (vault_address, _) = Pubkey::find_program_address(
            &[
                &market.address.to_bytes(),
                &token_mint.to_bytes(),
                b"vault",
            ],
            &spl_token::ID,
        );
        
        Ok(vault_address)
    }
}