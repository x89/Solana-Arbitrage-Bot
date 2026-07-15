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

pub struct Orca {
    pub rpc_client: RpcClient,
    pub program_id: Pubkey,
    pub authority: Keypair,
}

impl Orca {
    pub fn new(rpc_client: RpcClient, program_id: Pubkey, authority: Keypair) -> Self {
        Orca {
            rpc_client,
            program_id,
            authority,
        }
    }
}

#[async_trait]
impl DexTrait for Orca {
    async fn get_markets(&self) -> Result<Vec<Market>> {
        let mut markets = Vec::new();
        
        let pools = self.rpc_client.get_program_accounts(&self.program_id)?;
        
        for pool in pools {
            let pool_data: PoolData = bincode::deserialize(&pool.account.data)?;
            
            let market = Market {
                address: pool.pubkey,
                name: format!("{}/{}", pool_data.token_a.to_string(), pool_data.token_b.to_string()),
                base_asset: pool_data.token_a,
                quote_asset: pool_data.token_b,
                base_decimals: pool_data.token_a_decimals,
                quote_decimals: pool_data.token_b_decimals,
            };
            markets.push(market);
        }
        
        Ok(markets)
    }

    async fn get_orderbook(&self, market: &Market) -> Result<(Vec<Order>, Vec<Order>)> {
        Ok((Vec::new(), Vec::new()))
    }

    async fn place_order(
        &self,
        market: &Market,
        order_type: OrderType,
        side: OrderSide,
        price: f64,
        quantity: f64,
    ) -> Result<Order> {
        let pool_data = self.get_pool_data(&market.address).await?;
        
        let (token_a_amount, token_b_amount) = match side {
            OrderSide::Bid => (quantity, quantity * price),
            OrderSide::Ask => (quantity / price, quantity),
        };
        
        let minimum_amount_out = match side {
            OrderSide::Bid => token_b_amount * 0.99,
            OrderSide::Ask => token_a_amount * 0.99,
        };
        
        let instruction = orca_swap::instruction::swap(
            &self.program_id,
            &market.address,
            &self.authority.pubkey(),
            &pool_data.token_a_account,
            &pool_data.token_b_account,
            &self.get_token_account(&market.base_asset).await?,
            &self.get_token_account(&market.quote_asset).await?,
            token_a_amount,
            minimum_amount_out,
        )?;
        
        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        let transaction = Transaction::new_signed_with_payer(
            &[instruction],
            Some(&self.authority.pubkey()),
            &[&self.authority],
            recent_blockhash,
        );
        
        self.rpc_client.send_and_confirm_transaction(&transaction)?;
        
        Ok(Order {
            id: self.create_order_id(),
            market: market.clone(),
            order_type,
            side,
            price,
            quantity,
            status: OrderStatus::Filled,
        })
    }

    async fn cancel_order(&self, _order: &Order) -> Result<()> {
        Ok(())
    }

    async fn get_balances(&self, market: &Market) -> Result<HashMap<Pubkey, f64>> {
        let pool_data = self.get_pool_data(&market.address).await?;
        
        let token_a_balance = self.rpc_client.get_token_account_balance(&pool_data.token_a_account)?;
        let token_b_balance = self.rpc_client.get_token_account_balance(&pool_data.token_b_account)?;
        
        let mut balances = HashMap::new();
        balances.insert(market.base_asset, token_a_balance.amount as f64);
        balances.insert(market.quote_asset, token_b_balance.amount as f64);
        
        Ok(balances)
    }
}

impl Orca {
    fn create_order_id(&self) -> u64 {
        rand::random()
    }

    async fn get_pool_data(&self, pool_address: &Pubkey) -> Result<PoolData> {
        let pool_account_info = self.rpc_client.get_account(pool_address)?;
        let pool_data: PoolData = bincode::deserialize(&pool_account_info.data)?;
        
        Ok(pool_data)
    }

    async fn get_token_account(&self, token_mint: &Pubkey) -> Result<Pubkey> {
        let token_account = spl_associated_token_account::get_associated_token_address(
            &self.authority.pubkey(),
            token_mint,
        );
        
        Ok(token_account)
    }
}