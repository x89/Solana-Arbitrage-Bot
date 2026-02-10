use anchor_client::solana_client::rpc_client::RpcClient;
use anchor_client::solana_sdk::commitment_config::CommitmentConfig;
use anchor_client::solana_sdk::pubkey::Pubkey;
use anchor_client::solana_sdk::signature::read_keypair_file;
use anchor_client::solana_sdk::signature::{Keypair, Signer};

use anchor_client::{Client, Cluster};

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::rc::Rc;
use std::str::FromStr;

use std::borrow::Borrow;
use std::vec;

use clap::Parser;

use log::{debug, info, warn};
use solana_sdk::account::Account;

use client::arb::*;
use client::constants::*;
use client::pool::{pool_factory, PoolDir, PoolOperations, PoolType};
use client::serialize::token::unpack_token_account;
use client::utils::{
    derive_token_address, read_json_dir, PoolEdge, PoolGraph, PoolIndex, PoolQuote,
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    #[clap(short, long)]
    pub cluster: String,
}

fn add_pool_to_graph<'a>(
    graph: &mut PoolGraph,
    idx0: PoolIndex,
    idx1: PoolIndex,
    quote: &PoolQuote,
) {
    // idx0 = A, idx1 = B
    let edges = graph
        .0
        .entry(idx0)
        .or_insert_with(|| PoolEdge(HashMap::new()));
    let quotes = edges.0.entry(idx1).or_insert_with(|| vec![]);
    quotes.push(quote.clone());
}

fn main() {
    let args = Args::parse();
    let cluster = match args.cluster.as_str() {
        "localnet" => Cluster::Localnet,
        "mainnet" => Cluster::Mainnet,
        _ => panic!("invalid cluster type"),
    };

    env_logger::init();

    let owner_kp_path = match cluster {
        Cluster::Localnet => "../../mainnet_fork/localnet_owner.key",
        Cluster::Mainnet => {
            "/Users/edgar/.config/solana/uwuU3qc2RwN6CpzfBAhg6wAxiEx138jy5wB3Xvx18Rw.json"
        }
        _ => panic!("shouldnt get here"),
    };

    // ** setup RPC connection
    let connection_url = match cluster {
        Cluster::Mainnet => {
            "https://mainnet.rpc.jito.wtf/?access-token=746bee55-1b6f-4130-8347-5e1ea373333f"
        }
        _ => cluster.url(),
    };
    info!("using connection: {}", connection_url);

    let connection = RpcClient::new_with_commitment(connection_url, CommitmentConfig::confirmed());
    let send_tx_connection =
        RpcClient::new_with_commitment(cluster.url(), CommitmentConfig::confirmed());

    // setup anchor things
    let owner = read_keypair_file(owner_kp_path.clone()).unwrap();
    let rc_owner = Rc::new(owner);
    let provider = Client::new_with_options(
        cluster.clone(),
        rc_owner.clone(),
        CommitmentConfig::confirmed(),
    );
    let program = provider.program(*ARB_PROGRAM_ID);

    // ** define pool JSONs
    let mut pool_dirs = vec![];

    let orca_dir = PoolDir {
        tipe: PoolType::OrcaPoolType,
        dir_path: "../pools/orca".to_string(),
    };
    pool_dirs.push(orca_dir);

    let mercurial_dir = PoolDir {
        tipe: PoolType::MercurialPoolType,
        dir_path: "../pools/mercurial".to_string(),
    };
    pool_dirs.push(mercurial_dir);

    let saber_dir = PoolDir {
        tipe: PoolType::SaberPoolType,
        dir_path: "../pools/saber/".to_string(),
    };
    pool_dirs.push(saber_dir);

    // ** json pool -> pool object
    let mut token_mints = vec![];
    let mut pools = vec![];

    let mut update_pks = vec![];
    let mut update_pks_lengths = vec![];
    let mut all_mint_idxs = vec![];

    let mut mint2idx = HashMap::new();
    let mut graph_edges = vec![];

    info!("extracting pool + mints...");
    for pool_dir in pool_dirs {
        debug!("pool dir: {:#?}", pool_dir);
        let pool_paths = read_json_dir(&pool_dir.dir_path);

        for pool_path in pool_paths {
            let json_str = std::fs::read_to_string(&pool_path).unwrap();
            let pool = pool_factory(&pool_dir.tipe, &json_str);

            let pool_mints = pool.get_mints();
            if pool_mints.len() != 2 {
                // only support 2 mint pools
                warn!("skipping pool with mints != 2: {:?}", pool_path);
                continue;
            }

            //  ** record pool info for graph
            // token: (mint = graph idx), (addr = get quote amount)
            let mut mint_idxs = vec![];
            for mint in pool_mints {
                let idx;
                if !token_mints.contains(&mint) {
                    idx = token_mints.len();
                    mint2idx.insert(mint, idx);
                    token_mints.push(mint);
                    // graph_edges[idx] will always exist :)
                    graph_edges.push(HashSet::new());
                } else {
                    idx = *mint2idx.get(&mint).unwrap();
                }
                mint_idxs.push(idx);
            }

            // get accounts which need account info to be updated (e.g. pool src/dst amounts for xy=k)
            let update_accounts = pool.get_update_accounts();
            update_pks_lengths.push(update_accounts.len());
            update_pks.push(update_accounts);

            let mint0_idx = mint_idxs[0];
            let mint1_idx = mint_idxs[1];

            all_mint_idxs.push(mint0_idx);
            all_mint_idxs.push(mint1_idx);

            // record graph edges if they dont already exist
            if !graph_edges[mint0_idx].contains(&mint1_idx) {
                graph_edges[mint0_idx].insert(mint1_idx);
            }
            if !graph_edges[mint1_idx].contains(&mint0_idx) {
                graph_edges[mint1_idx].insert(mint0_idx);
            }

            pools.push(pool);
        }
    }
    let mut update_pks = update_pks.concat();

    info!("added {:?} mints", token_mints.len());
    info!("added {:?} pools", pools.len());

    // !
    let usdc_mint = Pubkey::from_str("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v").unwrap();
    let start_mint = usdc_mint;
    let start_mint_idx = *mint2idx.get(&start_mint).unwrap();

    let owner: &Keypair = rc_owner.borrow();
    let owner_start_addr = derive_token_address(&owner.pubkey(), &start_mint);

    // slide it in there
    update_pks.push(owner_start_addr);

    info!("getting pool amounts...");
    let mut update_accounts = vec![];
    for token_addr_chunk in update_pks.chunks(99) {
        let accounts = connection.get_multiple_accounts(token_addr_chunk).unwrap();
        update_accounts.push(accounts);
    }
    let mut update_accounts = update_accounts
        .concat()
        .into_iter()
        .filter(|s| s.is_some())
        .collect::<Vec<Option<Account>>>();

    info!("update accounts is {:?}", update_accounts.len());
    // slide it out here
    println!("accounts: {:#?}", update_accounts.clone());
    let init_token_acc = update_accounts.pop().unwrap().unwrap();
    let init_token_balance = unpack_token_account(&init_token_acc.data).amount as u128;
    info!(
        "init token acc: {:?}, balance: {:#}",
        init_token_acc, init_token_balance
    );
    info!("starting balance = {}", init_token_balance);

    info!("setting up exchange graph...");
    let mut graph = PoolGraph::new();
    let mut pool_count = 0;
    let mut account_ptr = 0;

    for pool in pools.into_iter() {
        // update pool
        let length = update_pks_lengths[pool_count];
        let _account_slice = &update_accounts[account_ptr..account_ptr + length].to_vec();
        account_ptr += length;

        // pool.set_update_accounts(*account_slice);

        // add pool to graph
        let idxs = &all_mint_idxs[pool_count * 2..(pool_count + 1) * 2].to_vec();
        let idx0 = PoolIndex(idxs[0]);
        let idx1 = PoolIndex(idxs[1]);

        let mut pool_ptr = PoolQuote::new(Rc::new(pool));
        add_pool_to_graph(&mut graph, idx0, idx1, &mut pool_ptr.clone());
        add_pool_to_graph(&mut graph, idx1, idx0, &mut pool_ptr);

        pool_count += 1;
    }

    let arbitrager = Arbitrager {
        token_mints,
        graph_edges,
        graph,
        cluster,
        owner: rc_owner,
        program,
        connection: send_tx_connection,
    };

    info!("searching for arbitrages...");
    let min_swap_amount = 10_u128.pow(6_u32); // scaled! -- 1 USDC
    let mut swap_start_amount = init_token_balance; // scaled!
    let mut sent_arbs = HashSet::new(); // track what arbs we did with a larger size

    for _ in 0..4 {
        arbitrager.brute_force_search(
            start_mint_idx,
            swap_start_amount,
            swap_start_amount,
            vec![start_mint_idx],
            vec![],
            &mut sent_arbs,
        );

        swap_start_amount /= 2; // half input amount and search again
        if swap_start_amount < min_swap_amount {
            break;
        } // dont get too small
    }
}