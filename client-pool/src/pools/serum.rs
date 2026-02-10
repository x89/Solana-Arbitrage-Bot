use core::panic;

use std::collections::HashMap;
use std::fmt::Debug;
use serde;
use serde::{Deserialize, Serialize};
use crate::pool::PoolOperations;
use crate::serialize::token::{WrappedPubkey};

use crate::utils::{derive_token_address}; 

use solana_sdk::pubkey::Pubkey;

use anchor_spl::dex::serum_dex::{
    critbit::{SlabView},
    matching::OrderBookState,
    state::Market,
};
use std::ops::DerefMut;


use anchor_client::{Program, Cluster};
use solana_sdk::instruction::Instruction;

use solana_sdk::clock::Epoch;
use solana_sdk::account::Account;
use solana_sdk::account_info::AccountInfo;
use crate::constants::*;
use crate::pool_utils::serum::*;

use anchor_spl::dex::serum_dex::{
    matching::Side,
};

use std::str::FromStr;
use tmp::accounts as tmp_accounts;
use tmp::instruction as tmp_instructions;

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SerumPool {
    pub own_address: WrappedPubkey,
    pub base_mint: WrappedPubkey,
    pub quote_mint: WrappedPubkey,
    pub base_scale: u64,
    pub quote_scale: u64,
    pub base_vault: WrappedPubkey,
    pub quote_vault: WrappedPubkey,
    pub request_queue: WrappedPubkey,
    pub event_queue: WrappedPubkey,
    pub bids: WrappedPubkey,
    pub asks: WrappedPubkey,
    pub vault_signer: WrappedPubkey,
    pub taker_fee_pct: f64,
    // !! 
    #[serde(skip)]
    pub accounts: Option<Vec<Option<Account>>>,
    #[serde(skip)]
    pub open_orders: Option<HashMap<String, String>>
}

fn account_info<'a>(pk: &'a Pubkey, account: &'a mut Account) -> AccountInfo<'a> {
    AccountInfo::new(
        pk, 
        false, 
        true, 
        &mut account.lamports, 
        &mut account.data, 
        &account.owner,
        false, 
        Epoch::default(),
    )
}

struct Iteration { 
    amount_in: u64, 
    amount_out: u64,
}

// bid: quote -> base 
fn bid_iteration(
    iteration: &mut Iteration,
    fee_tier: &FeeTier, 
    ob: &mut OrderBookState,
) -> bool {
    let quote_lot_size = ob.market_state.pc_lot_size;
    let base_lot_size = ob.market_state.coin_lot_size;

    let start_amount_in = iteration.amount_in;
    let max_pc_qty = fee_tier.remove_taker_fee(iteration.amount_in) / quote_lot_size;
    let mut pc_qty_remaining = max_pc_qty; 

    let done = loop {
        let flag = match ob.asks.find_min() { // min = best ask 
            Some(_) => false, 
            None => true
        };
        if flag { break true; }
        let best_ask = ob.asks.find_min().unwrap(); 
        let best_offer_ref = ob.asks.get_mut(best_ask).unwrap().as_leaf_mut().unwrap();
     
        let trade_price = best_offer_ref.price();
        let offer_size = best_offer_ref.quantity();
        let trade_qty = offer_size
            .min(pc_qty_remaining / best_offer_ref.price().get());

        if trade_qty == 0 { // fin 
            break true;
        }

        pc_qty_remaining -= trade_qty * trade_price.get();
        iteration.amount_out += trade_qty * base_lot_size; 

        best_offer_ref.set_quantity(best_offer_ref.quantity() - trade_qty);

        if best_offer_ref.quantity() == 0 {
            let best_offer_id = best_offer_ref.order_id();
            ob.asks.remove_by_key(best_offer_id)
                .unwrap();
        }
        break false; 
    };

    let native_accum_fill_price = (max_pc_qty - pc_qty_remaining) * quote_lot_size;
    let native_taker_fee = fee_tier.taker_fee(native_accum_fill_price);
    let native_pc_qty_remaining =
        start_amount_in - native_accum_fill_price - native_taker_fee;
    iteration.amount_in = native_pc_qty_remaining; 

    done
}

// ask: base -> quote 
fn ask_iteration(
    iteration: &mut Iteration,
    fee_tier: &FeeTier, 
    ob: &mut OrderBookState,
) -> bool {
    let pc_lot_size = ob.market_state.pc_lot_size;
    let coin_lot_size = ob.market_state.coin_lot_size;

    let max_qty = iteration.amount_in; 
    let mut unfilled_qty = max_qty / coin_lot_size;
    let mut accum_fill_price = 0;

    let done = loop {
        let best_bid = match ob.bids.find_max() { // min = best ask 
            Some(best_bid) => {
                best_bid
            }, 
            None => {
                break true; // no more bids
            }
        };
        let best_bid_ref = ob.bids.get_mut(best_bid).unwrap().as_leaf_mut().unwrap();
     
        let trade_price = best_bid_ref.price();
        let bid_size = best_bid_ref.quantity();
        let trade_qty = bid_size.min(unfilled_qty);

        if trade_qty == 0 { // fin 
            break true;
        }

        best_bid_ref.set_quantity(best_bid_ref.quantity() - trade_qty);
        unfilled_qty -= trade_qty;
        accum_fill_price += trade_qty * trade_price.get();

        if best_bid_ref.quantity() == 0 {
            let best_offer_id = best_bid_ref.order_id();
            ob.bids.remove_by_key(best_offer_id)
                .unwrap();
        }
        break false; 
    };
    // fees applied after
    let native_taker_pc_qty = accum_fill_price * pc_lot_size;
    let native_taker_fee = fee_tier.taker_fee(native_taker_pc_qty);
    let net_taker_pc_qty = native_taker_pc_qty - native_taker_fee;

    iteration.amount_out += net_taker_pc_qty;
    iteration.amount_in = unfilled_qty * coin_lot_size; 

    done
}

impl PoolOperations for SerumPool {

    fn get_name(&self) -> String {
        "Serum".to_string()
    }

    fn get_update_accounts(&self) -> Vec<Pubkey> {
        vec![
            self.own_address.0, 
            self.bids.0, 
            self.asks.0,
        ]
    }

    fn set_update_accounts(
        &mut self, 
        accounts: Vec<Option<Account>>,
        cluster: Cluster,
    ) {
        self.accounts = Some(accounts);
        let oo_path = match cluster { 
            Cluster::Localnet => {
                "./serum_open_orders.json"
            }, 
            Cluster::Mainnet => {
                panic!("TODO"); // idk 
                "./serum_open_orders.json"
            },
            _ => panic!("clsuter {} not supported", cluster)
        };
        let oo_str = std::fs::read_to_string(oo_path).unwrap();
        let oo_book: HashMap<String, String> = serde_json::from_str(&oo_str).unwrap();
        self.open_orders = Some(oo_book); 
    }

    fn mint_2_addr(&self, _mint: &Pubkey) -> Pubkey {
        panic!("ahhh")
    }

    fn get_mints(&self) -> Vec<Pubkey> {
        let mut mints = vec![
            self.base_mint.0,
            self.quote_mint.0
        ];
        mints.sort();
        mints
    }

    fn mint_2_scale(&self, mint: &Pubkey) -> u64 {
        if *mint == self.base_mint.0 {
            self.base_scale
        } else if *mint == self.quote_mint.0 { 
            self.quote_scale
        } else { 
            panic!("invalid mint bro")
        }
    }

    fn get_quote_with_amounts_scaled(
        &self, 
        amount_in: u128, 
        mint_in: &Pubkey,
        _mint_out: &Pubkey,
    ) -> u128 {

        let market_pk = self.own_address.0; 
        let fee_tier = FeeTier::from_srm_and_msrm_balances(&market_pk, 0, 0);
        let mut iteration = Iteration {
            amount_in: amount_in as u64,
            amount_out: 0,
        };

        let market_acc = &self.accounts.as_ref().unwrap()[0];
        let bids_acc = &self.accounts.as_ref().unwrap()[1];
        let asks_acc = &self.accounts.as_ref().unwrap()[2];
        
        // clone accounts for simulation (improve later?)
        let market_acc = &mut market_acc.clone().unwrap();
        let bid_acc = &mut bids_acc.clone().unwrap();
        let ask_acc = &mut asks_acc.clone().unwrap();

        let market_acc_info = &account_info(&self.own_address.0, market_acc);
        let bids_acc = &account_info(&self.bids.0, bid_acc);
        let asks_acc = &account_info(&self.asks.0, ask_acc);

        let mut market = Market::load(
            market_acc_info, 
            &SERUM_PROGRAM_ID
        ).unwrap();
        let mut bids = market.load_bids_mut(bids_acc).unwrap();
        let mut asks = market.load_asks_mut(asks_acc).unwrap();

        let mut ob = OrderBookState {
            bids: bids.deref_mut(),
            asks: asks.deref_mut(),
            market_state: market.deref_mut(),
        };
        
        if *mint_in == self.quote_mint.0 {
            // bid: quote -> base
            loop {
                let done = bid_iteration(
                    &mut iteration, 
                    &fee_tier, 
                    &mut ob, 
                );
                if done { break; }
            }
            iteration.amount_out as u128

        } else if *mint_in == self.base_mint.0 {
            // ask: base -> quote
            loop {
                let done = ask_iteration(
                    &mut iteration, 
                    &fee_tier, 
                    &mut ob, 
                );
                if done { break; }
            }
            iteration.amount_out as u128

        } else { 
            panic!("invalid mints");
        }
    }

    fn swap_ix(&self, 
        program: &Program,
        owner: &Pubkey,
        mint_in: &Pubkey, 
        _mint_out: &Pubkey
    ) -> Vec<Instruction> {

        let oos = self.open_orders.as_ref().unwrap(); 
        let open_orders = Pubkey::from_str(
            oos.get(&self.own_address.0.to_string()).unwrap()
        ).unwrap();

        let (swap_state, _) = Pubkey::find_program_address(
            &[b"swap_state"], 
            &program.id()
        );

        let base_ata = derive_token_address(owner, &self.base_mint);
        let quote_ata = derive_token_address(owner, &self.quote_mint);
        
        let side = if *mint_in == self.quote_mint.0 { Side::Bid }  else { Side::Ask };
        let payer_acc = if side == Side::Ask { base_ata } else { quote_ata };
        let _side = if side == Side::Ask { tmp::Side::Ask } else { tmp::Side::Bid };

        let request = program.request()
            .accounts(tmp_accounts::SerumSwap {
                market: tmp_accounts::MarketAccounts {
                    market: self.own_address.0, 
                    request_queue: self.request_queue.0,
                    event_queue: self.event_queue.0,
                    bids: self.bids.0,
                    asks: self.asks.0,
                    coin_vault: self.base_vault.0,
                    pc_vault: self.quote_vault.0,
                    vault_signer: self.vault_signer.0, 
                    open_orders,
                    order_payer_token_account: payer_acc, 
                    coin_wallet: base_ata, 
                },
                pc_wallet: quote_ata, 
                authority: *owner, 
                dex_program: *SERUM_PROGRAM_ID, 
                token_program: *TOKEN_PROGRAM_ID,
                rent: solana_sdk::sysvar::rent::id(),
                swap_state,
            })
            .args(tmp_instructions::SerumSwap { side: _side });

         

        request
            .instructions().unwrap()
    }

    fn can_trade(&self, 
        mint_in: &Pubkey,
        _mint_out: &Pubkey
    ) -> bool {

        let market_acc = &self.accounts.as_ref().unwrap()[0];
        let bids_acc = &self.accounts.as_ref().unwrap()[1];
        let asks_acc = &self.accounts.as_ref().unwrap()[2];
        
        // clone accounts for simulation (improve later?)
        let market_acc = &mut market_acc.clone().unwrap();
        let bid_acc = &mut bids_acc.clone().unwrap();
        let ask_acc = &mut asks_acc.clone().unwrap();

        let market_acc_info = &account_info(&self.own_address.0, market_acc);
        let bids_acc = &account_info(&self.bids.0, bid_acc);
        let asks_acc = &account_info(&self.asks.0, ask_acc);

        let market = Market::load(
            market_acc_info, 
            &SERUM_PROGRAM_ID
        ).unwrap();
        let bids = market.load_bids_mut(bids_acc).unwrap();
        let asks = market.load_asks_mut(asks_acc).unwrap();

        // is there a bid or ask we can trade with??? 
        if *mint_in == self.quote_mint.0 {
            // bid: quote -> base
            match asks.find_min() { // min = best ask 
                Some(_) => true, 
                None => false
            }
        } else if *mint_in == self.base_mint.0 {
            // ask: base -> quote
            match bids.find_max() {
                Some(_) => true, 
                None => false
            }

        } else { 
            panic!("invalid mints");
        }
    }
}