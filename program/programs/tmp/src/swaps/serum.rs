use anchor_lang::prelude::*;
use anchor_lang::Accounts;

use anchor_spl::dex;
use anchor_spl::dex::serum_dex::instruction::SelfTradeBehavior;
use anchor_spl::dex::serum_dex::matching::{OrderType, Side as SerumSide};
use anchor_spl::dex::serum_dex::state::MarketState;
use anchor_spl::token::TokenAccount;
use std::num::NonZeroU64;

use crate::state::SwapState;

// all taken from the serum swap program 

/// Convenience API to initialize an open orders account on the Serum DEX.
pub fn _init_open_order<'info>(ctx: Context<InitOpenOrder>) -> Result<()> {
    let ctx = CpiContext::new(ctx.accounts.dex_program.clone(), ctx.accounts.into());
    dex::init_open_orders(ctx).unwrap();
    Ok(())
}

pub fn _serum_swap<'info>(
    ctx: &Context<'_, '_, '_, 'info, SerumSwap<'info>>,
    amount_in: u64,
    side: Side,
) -> Result<()> {

    // Optional referral account (earns a referral fee).
    let referral = ctx.remaining_accounts.iter().next().map(Clone::clone);

    // Execute trade.
    let orderbook: OrderbookClient<'info> = (&*ctx.accounts).into();
    match side {
        Side::Bid => orderbook.buy(amount_in, None)?,
        Side::Ask => orderbook.sell(amount_in, None)?,
    };
    orderbook.settle(referral)?; // instant settle 

    Ok(())
}


#[derive(Accounts)]
pub struct InitOpenOrder<'info> {
    #[account(mut)]
    pub open_orders: AccountInfo<'info>,
    #[account(signer)]
    pub authority: AccountInfo<'info>,
    pub market: AccountInfo<'info>,
    pub dex_program: AccountInfo<'info>,
    pub rent: AccountInfo<'info>,
}

impl<'info> From<&mut InitOpenOrder<'info>> for dex::InitOpenOrders<'info> {
    fn from(accs: &mut InitOpenOrder<'info>) -> dex::InitOpenOrders<'info> {
        dex::InitOpenOrders {
            open_orders: accs.open_orders.clone(),
            authority: accs.authority.clone(),
            market: accs.market.clone(),
            rent: accs.rent.clone(),
        }
    }
}

#[derive(Accounts)]
pub struct CloseAccount<'info> {
    #[account(mut)]
    open_orders: AccountInfo<'info>,
    #[account(signer)]
    authority: AccountInfo<'info>,
    #[account(mut)]
    destination: AccountInfo<'info>,
    market: AccountInfo<'info>,
    dex_program: AccountInfo<'info>,
}

impl<'info> From<&mut CloseAccount<'info>> for dex::CloseOpenOrders<'info> {
    fn from(accs: &mut CloseAccount<'info>) -> dex::CloseOpenOrders<'info> {
        dex::CloseOpenOrders {
            open_orders: accs.open_orders.clone(),
            authority: accs.authority.clone(),
            destination: accs.destination.clone(),
            market: accs.market.clone(),
        }
    }
}

// The only constraint imposed on these accounts is that the market's base
// currency mint is not equal to the quote currency's. All other checks are
// done by the DEX on CPI.
#[derive(Accounts)]
pub struct SerumSwap<'info> {
    pub market: MarketAccounts<'info>,
    #[account(signer)]
    pub authority: AccountInfo<'info>,
    #[account(mut)]
    pub pc_wallet: Account<'info, TokenAccount>, // !! 
    // Programs.
    pub dex_program: AccountInfo<'info>,
    pub token_program: AccountInfo<'info>,
    // Sysvars.
    pub rent: AccountInfo<'info>,
    #[account(mut, seeds=[b"swap_state"], bump, )] 
    pub swap_state: Account<'info, SwapState>,
}

impl<'info> From<&SerumSwap<'info>> for OrderbookClient<'info> {
    fn from(accounts: &SerumSwap<'info>) -> OrderbookClient<'info> {
        OrderbookClient {
            market: accounts.market.clone(),
            authority: accounts.authority.clone(),
            pc_wallet: accounts.pc_wallet.to_account_info().clone(),
            dex_program: accounts.dex_program.clone(),
            token_program: accounts.token_program.clone(),
            rent: accounts.rent.clone(),
        }
    }
}

// Client for sending orders to the Serum DEX.
#[derive(Clone)]
struct OrderbookClient<'info> {
    market: MarketAccounts<'info>,
    authority: AccountInfo<'info>,
    pc_wallet: AccountInfo<'info>,
    dex_program: AccountInfo<'info>,
    token_program: AccountInfo<'info>,
    rent: AccountInfo<'info>,
}

impl<'info> OrderbookClient<'info> {
    // Executes the sell order portion of the swap, purchasing as much of the
    // quote currency as possible for the given `base_amount`.
    //
    // `base_amount` is the "native" amount of the base currency, i.e., token
    // amount including decimals.
    fn sell(
        &self,
        base_amount: u64,
        srm_msrm_discount: Option<AccountInfo<'info>>,
    ) -> Result<()> {
        let limit_price = 1;
        let max_coin_qty = {
            // The loaded market must be dropped before CPI.
            let market = MarketState::load(&self.market.market, &dex::ID).unwrap();
            coin_lots(&market, base_amount)
        };
        let max_native_pc_qty = u64::MAX;
        self.order_cpi(
            limit_price,
            max_coin_qty,
            max_native_pc_qty,
            Side::Ask,
            srm_msrm_discount,
        )
    }

    // Executes the buy order portion of the swap, purchasing as much of the
    // base currency as possible, for the given `quote_amount`.
    //
    // `quote_amount` is the "native" amount of the quote currency, i.e., token
    // amount including decimals.
    fn buy(
        &self,
        quote_amount: u64,
        srm_msrm_discount: Option<AccountInfo<'info>>,
    ) -> Result<()> {
        let limit_price = u64::MAX;
        let max_coin_qty = u64::MAX;
        let max_native_pc_qty = quote_amount;
        self.order_cpi(
            limit_price,
            max_coin_qty,
            max_native_pc_qty,
            Side::Bid,
            srm_msrm_discount,
        )
    }

    // Executes a new order on the serum dex via CPI.
    //
    // * `limit_price` - the limit order price in lot units.
    // * `max_coin_qty`- the max number of the base currency lot units.
    // * `max_native_pc_qty` - the max number of quote currency in native token
    //                         units (includes decimals).
    // * `side` - bid or ask, i.e. the type of order.
    // * `referral` - referral account, earning a fee.
    fn order_cpi(
        &self,
        limit_price: u64,
        max_coin_qty: u64,
        max_native_pc_qty: u64,
        side: Side,
        srm_msrm_discount: Option<AccountInfo<'info>>,
    ) -> Result<()> {
        // Client order id is only used for cancels. Not used here so hardcode.
        let client_order_id = 0;
        // Limit is the dex's custom compute budge parameter, setting an upper
        // bound on the number of matching cycles the program can perform
        // before giving up and posting the remaining unmatched order.
        let limit = 65535;

        let mut ctx = CpiContext::new(self.dex_program.clone(), self.clone().into());
        if let Some(srm_msrm_discount) = srm_msrm_discount {
            ctx = ctx.with_remaining_accounts(vec![srm_msrm_discount]);
        }
        dex::new_order_v3(
            ctx,
            side.into(),
            NonZeroU64::new(limit_price).unwrap(),
            NonZeroU64::new(max_coin_qty).unwrap(),
            NonZeroU64::new(max_native_pc_qty).unwrap(),
            SelfTradeBehavior::DecrementTake,
            OrderType::ImmediateOrCancel,
            client_order_id,
            limit,
        )
    }

    fn settle(&self, referral: Option<AccountInfo<'info>>) -> Result<()> {
        let settle_accs = dex::SettleFunds {
            market: self.market.market.clone(),
            open_orders: self.market.open_orders.clone(),
            open_orders_authority: self.authority.clone(),
            coin_vault: self.market.coin_vault.clone(),
            pc_vault: self.market.pc_vault.clone(),
            coin_wallet: self.market.coin_wallet.to_account_info().clone(),
            pc_wallet: self.pc_wallet.clone(),
            vault_signer: self.market.vault_signer.clone(),
            token_program: self.token_program.clone(),
        };
        let mut ctx = CpiContext::new(self.dex_program.clone(), settle_accs);
        if let Some(referral) = referral {
            ctx = ctx.with_remaining_accounts(vec![referral]);
        }
        dex::settle_funds(ctx)
    }
}

impl<'info> From<OrderbookClient<'info>> for dex::NewOrderV3<'info> {
    fn from(c: OrderbookClient<'info>) -> dex::NewOrderV3<'info> {
        dex::NewOrderV3 {
            market: c.market.market.clone(),
            open_orders: c.market.open_orders.clone(),
            request_queue: c.market.request_queue.clone(),
            event_queue: c.market.event_queue.clone(),
            market_bids: c.market.bids.clone(),
            market_asks: c.market.asks.clone(),
            order_payer_token_account: c.market.order_payer_token_account.clone(),
            open_orders_authority: c.authority.clone(),
            coin_vault: c.market.coin_vault.clone(),
            pc_vault: c.market.pc_vault.clone(),
            token_program: c.token_program.clone(),
            rent: c.rent.clone(),
        }
    }
}

// Returns the amount of lots for the base currency of a trade with `size`.
fn coin_lots(market: &MarketState, size: u64) -> u64 {
    size.checked_div(market.coin_lot_size).unwrap()
}

// Market accounts are the accounts used to place orders against the dex minus
// common accounts, i.e., program ids, sysvars, and the `pc_wallet`.
#[derive(Accounts, Clone)]
pub struct MarketAccounts<'info> {
    #[account(mut)]
    pub market: AccountInfo<'info>,
    #[account(mut)]
    pub open_orders: AccountInfo<'info>,
    #[account(mut)]
    pub request_queue: AccountInfo<'info>,
    #[account(mut)]
    pub event_queue: AccountInfo<'info>,
    #[account(mut)]
    pub bids: AccountInfo<'info>,
    #[account(mut)]
    pub asks: AccountInfo<'info>,
    // The `spl_token::Account` that funds will be taken from, i.e., transferred
    // from the user into the market's vault.
    //
    // For bids, this is the base currency. For asks, the quote.
    #[account(mut)]
    pub order_payer_token_account: AccountInfo<'info>,
    // Also known as the "base" currency. For a given A/B market,
    // this is the vault for the A mint.
    #[account(mut)]
    pub coin_vault: AccountInfo<'info>,
    // Also known as the "quote" currency. For a given A/B market,
    // this is the vault for the B mint.
    #[account(mut)]
    pub pc_vault: AccountInfo<'info>,
    // PDA owner of the DEX's token accounts for base + quote currencies.
    pub vault_signer: AccountInfo<'info>,
    // User wallets.
    #[account(mut)]
    pub coin_wallet: Account<'info, TokenAccount>,  // !! 
}

#[derive(AnchorSerialize, AnchorDeserialize)]
pub enum Side {
    Bid,
    Ask,
}

impl From<Side> for SerumSide {
    fn from(side: Side) -> SerumSide {
        match side {
            Side::Bid => SerumSide::Bid,
            Side::Ask => SerumSide::Ask,
        }
    }
}

// Event emitted when a swap occurs for two base currencies on two different
// markets (quoted in the same token).
#[event]
pub struct DidSwap {
    // User given (max) amount  of the "from" token to swap.
    pub given_amount: u64,
    // The minimum exchange rate for swapping `from_amount` to `to_amount` in
    // native units with decimals equal to the `to_amount`'s mint--specified
    // by the client.
    pub min_exchange_rate: ExchangeRate,
    // Amount of the `from` token sold.
    pub from_amount: u64,
    // Amount of the `to` token purchased.
    pub to_amount: u64,
    // The amount of the quote currency used for a *transitive* swap. This is
    // the amount *received* for selling on the first leg of the swap.
    pub quote_amount: u64,
    // Amount of the quote currency accumulated from a *transitive* swap, i.e.,
    // the difference between the amount gained from the first leg of the swap
    // (to sell) and the amount used in the second leg of the swap (to buy).
    pub spill_amount: u64,
    // Mint sold.
    pub from_mint: Pubkey,
    // Mint purchased.
    pub to_mint: Pubkey,
    // Mint of the token used as the quote currency in the two markets used
    // for swapping.
    pub quote_mint: Pubkey,
    // User that signed the transaction.
    pub authority: Pubkey,
}

// An exchange rate for swapping *from* one token *to* another.
#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct ExchangeRate {
    // The amount of *to* tokens one should receive for a single *from token.
    // This number must be in native *to* units with the same amount of decimals
    // as the *to* mint.
    pub rate: u64,
    // Number of decimals of the *from* token's mint.
    pub from_decimals: u8,
    // Number of decimals of the *to* token's mint.
    // For a direct swap, this should be zero.
    pub quote_decimals: u8,
    // True if *all* of the *from* currency sold should be used when calculating
    // the executed exchange rate.
    //
    // To perform a transitive swap, one sells on one market and buys on
    // another, where both markets are quoted in the same currency. Now suppose
    // one swaps A for B across A/USDC and B/USDC. Further suppose the first
    // leg swaps the entire *from* amount A for USDC, and then only half of
    // the USDC is used to swap for B on the second leg. How should we calculate
    // the exchange rate?
    //
    // If strict is true, then the exchange rate will be calculated as a direct
    // function of the A tokens lost and B tokens gained, ignoring the surplus
    // in USDC received. If strict is false, an effective exchange rate will be
    // used. I.e. the surplus in USDC will be marked at the exchange rate from
    // the second leg of the swap and that amount will be added to the
    // *to* mint received before calculating the swap's exchange rate.
    //
    // Transitive swaps only. For direct swaps, this field is ignored.
    pub strict: bool,
}

#[error_code]
pub enum SerumErrorCode {
    #[msg("The tokens being swapped must have different mints")]
    SwapTokensCannotMatch,
    #[msg("Slippage tolerance exceeded")]
    SlippageExceeded,
    #[msg("No tokens received when swapping")]
    ZeroSwap,
}
