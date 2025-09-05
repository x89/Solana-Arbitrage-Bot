use anchor_client::solana_client::rpc_client::RpcClient;
use anchor_client::solana_sdk::commitment_config::CommitmentConfig;
use anchor_client::solana_sdk::pubkey::Pubkey;
use anchor_client::solana_sdk::signature::{Keypair, Signer};
use anchor_client::solana_sdk::signature::read_keypair_file;
use anchor_client::{Client, Cluster};


use client::pools::SerumPool;

use solana_sdk::transaction::Transaction;

use std::rc::Rc;

use std::collections::{HashMap};

use std::vec;



use tmp::accounts as tmp_accounts;
use tmp::instruction as tmp_instructions;




use sha2::{Digest};

use anchor_lang::prelude::*;

use client::utils::{derive_token_address};
use client::pool::{ PoolType};
use client::constants::*;
use client::pool_utils::serum::*;

use solana_sdk::clock::Epoch;
use solana_sdk::account::Account;
use solana_sdk::account_info::AccountInfo;

use anchor_spl::dex::serum_dex::{
    critbit::{SlabView},
    matching::OrderBookState,
    state::Market,
};
use std::ops::DerefMut;

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

        println!("best ask {}", trade_price);

        if trade_qty == 0 { // fin 
            break true;
        }

        pc_qty_remaining -= trade_qty * trade_price.get();
        iteration.amount_out += trade_qty; 

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

fn main() {

    let cluster = Cluster::Localnet; 
    let connection = RpcClient::new_with_commitment(
        cluster.url(),
        CommitmentConfig::confirmed()
    );
    
    // let pool_path = "../../serum/serum_pools/ByRys5tuUWDgL73G8JBAEfkdFf8JWBzPBDHsBVQ5vbQA_serum_dex.json";
    // let pool_path = "../../serum/serum_pools/8PMHyKJ5FycCopijj6eXeCkenB71CYxCKH7AibkksdG5_serum_dex.json";
    let pool_path = "../../serum/serum_pools/7dLVkUfBVfCGkFhSXDCq1ukM9usathSgS716t643iFGF_serum_dex.json";
    let _pool_tipe = PoolType::SerumPoolType; 
    
    let contents = std::fs::read_to_string(pool_path).unwrap();
    let pool: SerumPool = serde_json::from_str(&contents).unwrap(); 

    // load market 
    let program_id = *SERUM_PROGRAM_ID;
    let market_pk = &pool.own_address.0; 
    let mut market_acc = connection.get_account(market_pk).unwrap();
    let market_info = account_info(market_pk, &mut market_acc);
    let mut market = Market::load(&market_info, &program_id).unwrap();

    // load bids + asks 
    let bids_pk = &pool.bids.0; 
    let asks_pk = &pool.asks.0;

    let bid_acc = connection.get_account(bids_pk).unwrap();

    let mut bid_acc_clone = bid_acc;
    let bid_info = account_info(bids_pk, &mut bid_acc_clone);
    let mut bids = market.load_bids_mut(&bid_info).unwrap();

    let ask_acc = connection.get_account(asks_pk).unwrap();
    let mut ask_acc_clone = ask_acc;
    let ask_info = account_info(asks_pk, &mut ask_acc_clone);
    let mut asks = market.load_asks_mut(&ask_info).unwrap();

    let mut ob = OrderBookState {
        bids: bids.deref_mut(),
        asks: asks.deref_mut(),
        market_state: market.deref_mut(),
    };

    let quote_scale = 10_u64.pow(pool.quote_scale as u32);
    let _base_scale = 10_u64.pow(pool.base_scale as u32);
    println!("{} {}", pool.quote_scale, pool.base_scale);
    
    let amount_in_u = 1000 * 10_u64.pow(pool.base_scale as u32); // 1 USDC  
    let fee_tier = FeeTier::from_srm_and_msrm_balances(market_pk, 0, 0);
    let mut iteration = Iteration {
        amount_in: amount_in_u,
        amount_out: 0,
    };

    loop {
        let done = bid_iteration(
        // let done = ask_iteration(
            &mut iteration, 
            &fee_tier, 
            &mut ob, 
        );
        if done { break; }
    }

    println!("amount_in residual: {}", iteration.amount_in);
    println!("QUOTE: amount out: {}", iteration.amount_out as f64 / quote_scale as f64);
    // println!("QUOTE: amount out: {}", ((iteration.amount_out * base_lot_size) as f64) / (base_scale as f64));
    println!("----");

    // drop(bids);
    // drop(asks);

    // // check -- currently only works if Account/Account-data is cloned after each 'simulation' 
    // let mut bid_acc_clone = bid_acc.clone();
    // let bid_info = account_info(&bids_pk, &mut bid_acc_clone);
    // let mut bids = market.load_bids_mut(&bid_info).unwrap();

    // let mut ask_acc_clone = ask_acc.clone();
    // let ask_info = account_info(&asks_pk, &mut ask_acc_clone);
    // let mut asks = market.load_asks_mut(&ask_info).unwrap();

    // let mut ob = OrderBookState {
    //     bids: bids.deref_mut(),
    //     asks: asks.deref_mut(),
    //     market_state: market.deref_mut(),
    // };
    // let mut iteration = Iteration {
    //     amount_in: amount_in_u,
    //     amount_out: 0,
    // };
    // loop {
    //     let done = bid_iteration(
    //         &mut iteration, 
    //         &fee_tier, 
    //         &mut ob, 
    //     );
    //     if done { break; }
    // }
    // println!("residual V2: {}", iteration.amount_in);
    // println!("QUOTE V2: amount out: {}", ((iteration.amount_out * size_ratio) as f64) / (base_scale as f64));
    // println!("----");

    // do a swap and check the amount         
    let mut PROGRAM_LAYOUT_VERSIONS = HashMap::new(); 
    PROGRAM_LAYOUT_VERSIONS.insert("4ckmDgGdxQoPDLUkDT3vHgSAkzA3QRdNq5ywwY4sUSJn", 1);
    PROGRAM_LAYOUT_VERSIONS.insert("BJ3jrUzddfuSrZHXSCxMUUQsjKEyLmuuyZebkcaFp2fg", 1);
    PROGRAM_LAYOUT_VERSIONS.insert("EUqojwWA2rd19FZrzeBncJsm38Jm1hEhE3zsmX3bRc2o", 2);
    PROGRAM_LAYOUT_VERSIONS.insert("9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin", 3);

    let LAYOUT_V1_SPAN = 3220; 
    let LAYOUT_V2_SPAN = 3228; 

    let layout_v = PROGRAM_LAYOUT_VERSIONS.get(SERUM_PROGRAM_ID.to_string().as_str()).unwrap();
    let space = if *layout_v == 1 { LAYOUT_V1_SPAN } else { LAYOUT_V2_SPAN };

    let owner_kp_path = "../../mainnet_fork/localnet_owner.key";   
    let owner = read_keypair_file(owner_kp_path.clone()).unwrap();
    println!("{}", owner.pubkey());
    
    let open_orders = Keypair::new(); 

    let rent_exemption_amount = connection
        .get_minimum_balance_for_rent_exemption(space)
        .unwrap();

    let create_account_ix = solana_sdk::system_instruction::create_account(
        &owner.pubkey(),
        &open_orders.pubkey(),
        rent_exemption_amount,
        space as u64,
        &program_id,
    );

    // setup anchor things 
    let provider = Client::new_with_options(
        cluster, 
        Rc::new(owner), 
        CommitmentConfig::confirmed() 
    );
    let program = provider.program(*ARB_PROGRAM_ID);
    let owner = read_keypair_file(owner_kp_path.clone()).unwrap();     

    let init_ix = program.request()
        .accounts(tmp_accounts::InitOpenOrder{
            open_orders: open_orders.pubkey(), 
            authority: owner.pubkey(), 
            market: pool.own_address.0, 
            dex_program: program_id,
            rent: solana_sdk::sysvar::rent::id(),
        })
        .args(tmp_instructions::InitOpenOrder {})
        .instructions()
        .unwrap();

    let side = "buy";
    
    let base_ata = derive_token_address(&owner.pubkey(), &pool.base_mint);
    let quote_ata = derive_token_address(&owner.pubkey(), &pool.quote_mint);

    let base_balance = connection.get_token_account_balance(&base_ata).unwrap();
    let quote_balance = connection.get_token_account_balance(&quote_ata).unwrap();

    println!("{:#?}", amount_in_u);
    println!("{:#?} {:#?}", base_balance.amount, quote_balance.amount);

    let payer_acc = if side == "buy" { quote_ata } else { base_ata };
    let _side = if side == "buy" { tmp::Side::Bid } else { tmp::Side::Ask };

    let (swap_state, _) = Pubkey::find_program_address(
        &[b"swap_state"], 
        &program.id()
    );

    // initialize swap 
    let ix = program
        .request()
        .accounts(tmp_accounts::TokenAndSwapState {
            swap_state,
            src: base_ata,
        })
        .args(tmp_instructions::StartSwap {
            swap_input: amount_in_u
        }).instructions().unwrap();

    let swap_ix = program.request()
        .accounts(tmp_accounts::SerumSwap {
            market: tmp_accounts::MarketAccounts {
                market: pool.own_address.0, 
                request_queue: pool.request_queue.0,
                event_queue: pool.event_queue.0,
                bids: pool.bids.0,
                asks: pool.asks.0,
                coin_vault: pool.base_vault.0,
                pc_vault: pool.quote_vault.0,
                vault_signer: pool.vault_signer.0, 
                open_orders: open_orders.pubkey(),
                order_payer_token_account: payer_acc, 
                coin_wallet: base_ata, 
            },
            pc_wallet: quote_ata, 
            authority: owner.pubkey(), 
            dex_program: program_id, 
            token_program: *TOKEN_PROGRAM_ID,
            rent: solana_sdk::sysvar::rent::id(),
            swap_state,
        })
        .args(tmp_instructions::SerumSwap { side: _side })
        .instructions().unwrap(); 

    let ixs = vec![
        vec![create_account_ix], 
        init_ix, 
        ix,
        swap_ix
    ].concat();

    let mut tx = Transaction::new_with_payer(
        &ixs, 
        Some(&owner.pubkey())
    );

    let base_balance_src = connection.get_token_account_balance(&base_ata).unwrap().amount.parse::<u64>().unwrap();
    let quote_balance_src = connection.get_token_account_balance(&quote_ata).unwrap().amount.parse::<u64>().unwrap();
    println!("base, quote balance: {} {}", base_balance_src, quote_balance_src);

    let recent_hash = connection.get_latest_blockhash().unwrap();
    tx.sign(&[&owner, &open_orders], recent_hash);
    
    // connection.send_and_confirm_transaction(&tx).unwrap();

    let resp = connection.simulate_transaction(&tx).unwrap();
    println!("{:#?}", resp);

    // let ixs2 = ixs;
    // let mut tx2 = Transaction::new_with_payer(
    //     &ixs2, 
    //     Some(&owner.pubkey())
    // );
    // let recent_hash = connection.get_latest_blockhash().unwrap();
    // tx2.sign(&[&owner], recent_hash);
    // let resp = connection.simulate_transaction(&tx2).unwrap();
    // println!("{:#?}", resp);

    let base_balance_dst = connection.get_token_account_balance(&base_ata).unwrap().amount.parse::<u64>().unwrap();
    let quote_balance_dst = connection.get_token_account_balance(&quote_ata).unwrap().amount.parse::<u64>().unwrap();
    println!("base, quote balance: {} {}", base_balance_dst, quote_balance_dst);

    if side == "buy" {
        println!("SWAP: quote_diff: -{} base_diff: {}", quote_balance_src - quote_balance_dst, base_balance_dst - base_balance_src);
    } else { 
        println!("SWAP: quote_diff: {} base_diff: -{}",  quote_balance_dst - quote_balance_src, base_balance_src - base_balance_dst);
    }

    // // aldrin AMM quotes = jupiter's quotes :) 
    // let src_amount = 295787369218 as u128; // USDC in 
    // let dst_amount = 282161482788394 as u128; // RIN out
    
    // let src_scale = 6; 
    // let dst_scale = 9; 

    // let trade_direction = AtoB;
    // let fees = Fees {
    //     trade_fee_numerator: 20,
    //     trade_fee_denominator: 10000,
    //     owner_trade_fee_numerator: 10,
    //     owner_trade_fee_denominator: 10000,
    //     owner_withdraw_fee_numerator: 0,
    //     owner_withdraw_fee_denominator: 0,
    //     host_fee_numerator: 0,
    //     host_fee_denominator: 0,
    // };

    // let amount_in = (1 * 10_u64.pow(src_scale)) as u128; 
    // let swap_curve = SwapCurve {
    //     curve_type: CurveType::ConstantProduct,
    //     calculator: Arc::new(ConstantProductCurve {}),
    // };
    // let quote = swap_curve.swap(
    //     amount_in, 
    //     src_amount, 
    //     dst_amount, 
    //     trade_direction, 
    //     &fees
    // ).unwrap().destination_amount_swapped;
    
    // let fquote = (quote as f64) / (10_u64.pow(dst_scale) as f64);
    // println!("{}", fquote);

    // // simulate a swap 
    // let pool_dir = "../../aldrin_sdk/pools/".to_string();
    // let pool_tipe = PoolType::AldrinPoolType; 
    
    // let cluster = Cluster::Localnet; // !!! make sure its localnet lmfao 
    // let connection = RpcClient::new_with_commitment(
    //     cluster.url(),
    //     CommitmentConfig::confirmed()
    // );

    // let pool_paths = read_json_dir(&pool_dir);
    // // let pool_path = &pool_paths[0];
    // // v2 pool explicit
    // let pool_path = "../../aldrin_sdk/pools/FZKYeYPqyJcjLdLEoUR3UrVpSzR46LB2Lop5dcNV2WZR_aldrin_pool.json";
    // println!("{:?}", pool_path);

    // let contents = std::fs::read_to_string(&pool_path).unwrap();
    // let _pool = pool_factory(&pool_tipe, &contents);
    // let pool = match &_pool {
    //     Pool::AldrinPool(pool) => pool, 
    //     _ => panic!("ahhh")
    // };

    // println!("{}", pool.pool_version);

    // let owner_kp_path = "../../../mainnet.key";   
    // let owner = read_keypair_file(owner_kp_path.clone()).unwrap();

    // let base_token_mint = &pool.token_ids[0];
    // let quote_token_mint = &pool.token_ids[1];

    // let mint_in = base_token_mint;
    // let mint_out = quote_token_mint;

    // let base_token_vault = pool.tokens
    //     .get(base_token_mint)
    //     .unwrap()
    //     .addr.0;
    // let quote_token_vault = pool.tokens
    //     .get(quote_token_mint)
    //     .unwrap()
    //     .addr.0;

    // // derive atas 
    // let user_base_ata = derive_token_address(
    //     &owner.pubkey(), 
    //     &Pubkey::from_str(base_token_mint).unwrap()
    // );
    // let user_quote_ata = derive_token_address(
    //     &owner.pubkey(), 
    //     &Pubkey::from_str(quote_token_mint).unwrap()
    // );

    // let prefix = "global".to_string();
    // let name = "swap".to_string();
    // let key = format!("{}:{}", prefix, name);
    // let mut hasher = Sha256::new(); 
    // hasher.update(key);
    // let result = hasher.finalize();
    // let fcn_name = &result.as_slice()[..8];

    // let amount_in = &(5000000 as u64).try_to_vec().unwrap()[..]; // input amount 
    // let mint_amount_out = &(4780 as u64).try_to_vec().unwrap()[..]; // min output? 

    // let is_inverted = mint_out == quote_token_mint;
    // let bid_ask_flag = if is_inverted { 1 } else { 0 }; // 0 = bid, 1 = ask 
    // let bid_ask = &[bid_ask_flag];

    // let data = [
    //     fcn_name, 
    //     amount_in, 
    //     mint_amount_out, 
    //     bid_ask, 
    // ].concat();

    // let mut keys = vec![
    //     AccountMeta::new_readonly(pool.pool_public_key.0, false),
    //     AccountMeta::new_readonly(pool.pool_signer.0, false),
    //     AccountMeta::new(pool.pool_mint.0, false),
    //     AccountMeta::new(base_token_vault, false), 
    //     AccountMeta::new(quote_token_vault, false), 
    //     AccountMeta::new(pool.fee_pool_token_account.0, false),
    //     AccountMeta::new_readonly(owner.pubkey().clone(), true),
    //     AccountMeta::new(user_base_ata, false), 
    //     AccountMeta::new(user_quote_ata, false), 
    // ]; 
    // let program_id; 
    // if pool.pool_version == 1 {
    //     program_id = str2pubkey(ALDRIN_V1_PROGRAM_ID);
    // } else {  // version 2! only 2nd last account changes 
    //     program_id = str2pubkey(ALDRIN_V2_PROGRAM_ID);
    //     keys.push(
    //         AccountMeta::new_readonly(pool.curve.0, false), 
    //     );
    // }
    // keys.push(
    //     AccountMeta::new_readonly(str2pubkey(TOKEN_PROGRAM_ID), false)
    // );

    // let ix = Instruction {
    //     program_id: program_id,
    //     accounts: keys,
    //     data: data.clone(),
    // };
    // let ixs = vec![ix];
    
    // let (rh, _fee_calc) = connection.get_recent_blockhash().unwrap();
    // let tx = Transaction::new_signed_with_payer(
    //     &ixs,
    //     Some(&owner.pubkey()),
    //     &[&owner],
    //     rh,
    // );

    // // let response = connection.simulate_transaction(&tx).unwrap();
    // // println!("{:?}", response);
    // // println!("----");

    // let owner = read_keypair_file(owner_kp_path.clone()).unwrap();
    // let provider = Client::new_with_options(
    //     cluster.clone(), 
    //     Rc::new(owner), 
    //     CommitmentConfig::confirmed() 
    // );
    // let owner = read_keypair_file(owner_kp_path.clone()).unwrap();
    // let program = provider.program(str2pubkey(ARB_PROGRAM_ID));

    // let (info_pda, _) = Pubkey::find_program_address(
    //     &[b"swap_state"], 
    //     &program.id()
    // );

    // // let is_inverted = mint_out == quote_token_mint;
    // // let dst_is_quote_ata = mint_out == quote_token_mint;

    // let src_scale = _pool.mint_2_scale(Pubkey::from_str(mint_in).unwrap());
    // let amount_in = 1 * 10_u64.pow(src_scale as u32);

    // let ix = program
    //     .request()
    //     .accounts(tmp_accounts::Information {
    //         information: info_pda,
    //     })
    //     .args(tmp_ix::StartSwap {
    //         new_amount: amount_in as u64
    //     }).instructions().unwrap();

    // let ixs_anc = program
    //     .request()
    //     .accounts(tmp_accounts::AldrinSwapV1 {
    //         pool_public_key: pool.pool_public_key.0,
    //         pool_signer: pool.pool_signer.0,
    //         pool_mint: pool.pool_mint.0,
    //         base_token_vault: base_token_vault, 
    //         quote_token_vault: quote_token_vault, 
    //         fee_pool_token_account: pool.fee_pool_token_account.0,
    //         user_transfer_authority: owner.pubkey(),
    //         user_base_ata: user_base_ata, 
    //         user_quote_ata: user_quote_ata,
    //         // ...
    //         aldrin_v1_program: str2pubkey(ALDRIN_V1_PROGRAM_ID),
    //         token_program: str2pubkey(TOKEN_PROGRAM_ID),
    //         information: info_pda, 
    //     })
    //     .args(tmp_ix::AldrinSwapV1 { is_inverted })
    //     .instructions()
    //     .unwrap();

    // let all_ixs = vec![ix, ixs_anc].concat();

    // let (rh, _fee_calc) = connection.get_recent_blockhash().unwrap();
    // let tx = Transaction::new_signed_with_payer(
    //     &all_ixs,
    //     Some(&owner.pubkey()),
    //     &[&owner],
    //     rh,
    // );
    // let response = connection.simulate_transaction(&tx).unwrap();
    // println!("{:?}", response);
    
    // // test out a mercurial swap instruction here bc of 
    // let d = SwapData {
    //     instruction: 4, // mercurial swap ix 
    //     amount_in: 2000000000,
    //     minimum_amount_out: 0,
    // };

    // let enc_d = d.try_to_vec().unwrap();
    // println!("{:?}", enc_d);

    // let mut graph_v2 = HashMap::new();

    // let mut hm = HashMap::new();
    // hm.insert(0, vec![0]);
    // graph_v2.insert(0, hm);

    // let mut hm2 = graph_v2.get_mut(&0).unwrap(); 
    // let mut v = hm2.get_mut(&0).unwrap();
    // v.push(2);

    // hm3.push(2);

    // let pool_directory = "/Users/brennan/Documents/workspace/solana/solana_penguin/mercurial_pools/pools/";
    // // import pool + token information
    // let all_pool_paths = {
    //     let _paths = fs::read_dir(pool_directory).unwrap();
    //     let mut paths = Vec::new();
    //     for path in _paths {
    //         let p = path.unwrap().path();
    //         let path_str = p;
    //         match path_str.extension() {
    //             Some(ex) => {
    //                 if ex == "json" {
    //                     paths.push(path_str);
    //                 }
    //             },
    //             None => {}
    //         }
    //     }
    //     paths
    // };

    // let mut merc_pools = vec![];
    // for pool_path in all_pool_paths {
    //     let pool_str = std::fs::read_to_string(pool_path.to_str().unwrap()).unwrap();
    //     let pool: MercurialPool = serde_json::from_str(&pool_str).unwrap(); 
    //     merc_pools.push(pool);
    // }

    // // TEST WITH SINGLE POOL 
    // let pool = &merc_pools[merc_pools.len()-1];
    // println!("{:?}", pool);

    // // let connection_url = "https://ssc-dao.genesysgo.net/";
    // let connection_url = "http://127.0.0.1:8899/";
    // let connection = RpcClient::new_with_commitment(
    //     connection_url,
    //     CommitmentConfig::confirmed()
    // );

    // let token_accounts: Vec<Pubkey> = pool.token_accounts
    //     .iter()
    //     .map(|pk| pk.0)
    //     .collect();
    // println!("{:?}", token_accounts);

    // let pool_token_accounts = connection
    //     .get_multiple_accounts(&token_accounts)
    //     .unwrap();

    // let mut token_amounts = vec![];
    // for account in pool_token_accounts {
    //     let data = account.unwrap().data;
    //     let amount = unpack_token_account(&data).amount as u128;
    //     token_amounts.push(amount);
    // }

    // let percision_multipliers = &pool.precision_multiplier;
    // let amp = pool.amp;
    // let input_idx = 1; 
    // let output_idx = 0; 
    
    // // scale by percision 
    // // let true_input_amount = 11457257;
    // // let source_amount = true_input_amount * 10_u128.pow(9 as u32); // input scale amount 
    // let source_amount = 1000000000;

    // let xp: Vec<u128> = token_amounts.iter().enumerate().map(|(i, amount)| {
    //     amount * percision_multipliers[i] as u128
    // }).collect();

    // // println!("{:?}", token_amounts); // [123419283340931, 60442114126290]
    // // println!("{:?}", percision_multipliers); // [1012, 1000, 0, 0]

    // let dx = source_amount * percision_multipliers[input_idx] as u128;

    // let x = xp[input_idx] + dx;
    // let leverage = compute_a(amp).unwrap();
    // let d = compute_d(leverage, xp[0], xp[1]).unwrap();
    // let y = compute_y(leverage, x, d).unwrap();
    // let dy = xp[output_idx] - y;
    // let out_amount = dy.checked_div(percision_multipliers[output_idx] as u128).unwrap();

    // // compute fees 
    // let fee_denom = 10_u128.pow(10); // from jupiter SDK 
    // let fees = out_amount
    //     .checked_mul(pool.fee_numerator as u128).unwrap()
    //     .checked_div(fee_denom).unwrap();
    // let out_amount_with_fees = out_amount - fees;
    
    // println!("{}", out_amount_with_fees);
}