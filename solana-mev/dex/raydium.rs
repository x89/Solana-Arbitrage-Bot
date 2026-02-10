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

pub struct Raydium {
    pub rpc_client: RpcClient,
    pub program_id: Pubkey,
    pub amm_id: Pubkey,
    pub serum_program_id: Pubkey,
    pub authority: Keypair,
}

impl Raydium {
    pub fn new(
        rpc_client: RpcClient,
        program_id: Pubkey,
        amm_id: Pubkey,
        serum_program_id: Pubkey,
        authority: Keypair,
    ) -> Self {
        Raydium {
            rpc_client,
            program_id,
            amm_id,
            serum_program_id,
            authority,
        }
    }
}

#[async_trait]
impl DexTrait for Raydium {
    async fn get_markets(&self) -> Result<Vec<Market>> {
        let mut markets = Vec::new();
        
        let market_infos = self.rpc_client.get_account_info(&self.amm_id)?;
        let market_data: AmmInfo = bincode::deserialize(&market_infos.data)?;
        
        for (mint_a, mint_b) in market_data.mints.iter() {
            let market = Market {
                address: Pubkey::default(),
                name: format!("{}/{}", mint_a, mint_b),
                base_asset: *mint_a,
                quote_asset: *mint_b,
                base_decimals: 0,
                quote_decimals: 0,
            };
            markets.push(market);
        }
        
        Ok(markets)
    }

    async fn get_orderbook(&self, market: &Market) -> Result<(Vec<Order>, Vec<Order>)> {
        let market_account_info = self.rpc_client.get_account_info(&market.address)?;
        let market_data: MarketState = bincode::deserialize(&market_account_info.data)?;
        
        let bids = market_data.bids.iter().map(|order| Order {
            price: order.price,
            quantity: order.quantity,
            side: OrderSide::Bid,
        }).collect();
        
        let asks = market_data.asks.iter().map(|order| Order {
            price: order.price,
            quantity: order.quantity,
            side: OrderSide::Ask,
        }).collect();
        
        Ok((bids, asks))
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
        
        let (vault_a, vault_b) = self.get_vaults(market).await?;
        
        let (token_a_account, token_b_account) = self.get_token_accounts(market).await?;
        
        let order_data = match side {
            OrderSide::Bid => MarketInstruction::NewOrder {
                order_type: order_type.into(),
                side: Side::Bid,
                limit_price: price,
                max_quantity: quantity,
                order_id,
            },
            OrderSide::Ask => MarketInstruction::NewOrder {
                order_type: order_type.into(),
                side: Side::Ask,
                limit_price: price,
                max_quantity: quantity,
                order_id,
            },
        };
        
        let accounts = match side {
            OrderSide::Bid => vec![
                AccountMeta::new(market.address, false),
                AccountMeta::new(order_account.pubkey(), true),
                AccountMeta::new(self.authority.pubkey(), true),
                AccountMeta::new_readonly(spl_token::ID, false),
                AccountMeta::new(token_b_account, false),
                AccountMeta::new(vault_b, false),
                AccountMeta::new(vault_a, false),
                AccountMeta::new(token_a_account, false),
                AccountMeta::new_readonly(solana_sdk::sysvar::rent::ID, false),
            ],
            OrderSide::Ask => vec![
                AccountMeta::new(market.address, false),
                AccountMeta::new(order_account.pubkey(), true),
                AccountMeta::new(self.authority.pubkey(), true),
                AccountMeta::new_readonly(spl_token::ID, false),
                AccountMeta::new(token_a_account, false),
                AccountMeta::new(vault_a, false),
                AccountMeta::new(vault_b, false),
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
        let cancel_data = MarketInstruction::CancelOrder { order_id: order.id };
        
        let accounts = vec![
            AccountMeta::new(order.market.address, false),
            AccountMeta::new_readonly(self.authority.pubkey(), true),
            AccountMeta::new(self.get_bids_address(&order.market)?, false),
            AccountMeta::new(self.get_asks_address(&order.market)?, false),
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
        let (vault_a, vault_b) = self.get_vaults(market).await?;
        
        let vault_a_balance = self.rpc_client.get_token_account_balance(&vault_a)?;
        let vault_b_balance = self.rpc_client.get_token_account_balance(&vault_b)?;
        
        let mut balances = HashMap::new();
        balances.insert(market.base_asset, vault_a_balance.amount as f64);
        balances.insert(market.quote_asset, vault_b_balance.amount as f64);
        
        Ok(balances)
    }
}

impl Raydium {
    fn create_order_id(&self) -> u64 {
        rand::random()
    }

    async fn get_vaults(&self, market: &Market) -> Result<(Pubkey, Pubkey)> {
        let market_account_info = self.rpc_client.get_account_info(&market.address)?;
        let market_data: MarketState = bincode::deserialize(&market_account_info.data)?;
        
        Ok((market_data.base_vault, market_data.quote_vault))
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
                &spl_token::ID.to_bytes(),
                &self.program_id.to_bytes(),
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
                &spl_token::ID.to_bytes(),
                &self.program_id.to_bytes(),
                b"asks",
            ],
            &self.program_id,
        );
        
        Ok(asks_address)
    }
}