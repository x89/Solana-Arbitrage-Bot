use anchor_client::solana_client::rpc_client::RpcClient;
use anchor_client::solana_sdk::commitment_config::CommitmentConfig;
use anchor_client::solana_sdk::pubkey::Pubkey;
use anchor_client::solana_sdk::signature::read_keypair_file;
use anchor_client::solana_sdk::signature::{Keypair, Signer};
use anchor_client::{Client, Cluster};
use clap::Parser;
use log::{debug, info, warn};
use reqwest::blocking::get;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::rc::Rc;
use std::str::FromStr;
use solana_sdk::account::Account;
use std::fs::File;
use std::io::{self, Read};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    #[clap(short, long)]
    pub cluster: String,
    #[clap(short, long, default_value = "config.json")]
    pub config: String,
}

#[derive(Debug, Deserialize)]
struct Config {
    fee_percentage: f64,
    dex_urls: Vec<String>,
}

fn fetch_tokens() -> Vec<String> {
    let response: Value = get("https://tokens.jup.ag/tokens?tags=birdeye-trending")
        .expect("Failed to fetch tokens")
        .json()
        .expect("Failed to parse JSON");

    response.as_array()
        .expect("Expected an array")
        .iter()
        .filter_map(|token| token.get("address").and_then(Value::as_str))
        .map(String::from)
        .collect()
}

fn add_pool_to_graph<'a>(
    graph: &mut PoolGraph,
    idx0: PoolIndex,
    idx1: PoolIndex,
    quote: &PoolQuote,
) {
    let edges = graph
        .0
        .entry(idx0)
        .or_insert_with(|| PoolEdge(HashMap::new()));
    let quotes = edges.0.entry(idx1).or_insert_with(|| vec![]);
    quotes.push(quote.clone());
}

fn calculate_fees(amount: u128, fee_percentage: f64) -> u128 {
    let fee = (amount as f64 * fee_percentage).round() as u128;
    fee
}

fn load_config(file_path: &str) -> Config {
    let mut file = File::open(file_path).expect("Could not open config file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Could not read config file");
    serde_json::from_str(&contents).expect("Could not parse config JSON")
}

fn main() {
        // Parse command-line arguments and initialize variables
    let args = Args::parse();
    let cluster = match args.cluster.as_str() {
        "localnet" => Cluster::Localnet,
        "mainnet" => Cluster::Mainnet,
        _ => panic!("invalid cluster type"),
    };
     // Initialize logging and other setup
    env_logger::init();
    // Fetch token mints from the blockchain
    let config = load_config(&args.config);
    
    let owner_kp_path = match cluster {
        Cluster::Localnet => "../../mainnet_fork/localnet_owner.key",
        Cluster::Mainnet => {
            "/Users/edgar/.config/solana/uwuU3qc2RwN6CpzfBAhg6wAxiEx138jy5wB3Xvx18Rw.json"
        }
        _ => panic!("shouldn't get here"),
    };

    let connection_url = match cluster {
        Cluster::Mainnet => {
            "https://mainnet.rpc.jito.wtf/?access-token=746bee55-1b6f-4130-8347-5e1ea373333f"
        }
        _ => cluster.url(),
    };
    info!("Using connection: {}", connection_url);

    let connection = RpcClient::new_with_commitment(connection_url, CommitmentConfig::confirmed());
    let send_tx_connection =
        RpcClient::new_with_commitment(cluster.url(), CommitmentConfig::confirmed());

    let owner = read_keypair_file(owner_kp_path.clone()).unwrap();
    let rc_owner = Rc::new(owner);
    let provider = Client::new_with_options(
        cluster.clone(),
        rc_owner.clone(),
        CommitmentConfig::confirmed(),
    );
    let program = provider.program(*ARB_PROGRAM_ID);

    let mut pool_dirs = vec![];

// Add DEX directories to the pool directories list
    let orca_dir = PoolDir {
        tipe: PoolType::OrcaPoolType,
        dir_path: "../pools/orca".to_string(),
    };
    pool_dirs.push(orca_dir);

    let raydium_dir = PoolDir {
        tipe: PoolType::SaberPoolType,
        dir_path: "../pools/raydium/".to_string(),
    };
    pool_dirs.push(saber_dir);

    let jupiter_dir = PoolDir {
        tipe: PoolType::JupiterPoolType,
        dir_path: "../pools/jupiter/".to_string(),
    };
    pool_dirs.push(saber_dir);

    // let meteora_dir = PoolDir {
    //     tipe: PoolType::SaberPoolType,
    //     dir_path: "../pools/meteora/".to_string(),
    // };
    // pool_dirs.push(saber_dir);

    // let phoenix_dir = PoolDir {
    //     tipe: PoolType::PhoenixPoolType,
    //     dir_path: "../pools/phoenix/".to_string(),
    // };
    // pool_dirs.push(saber_dir);

    // let lifinity_dir = PoolDir {
    //     tipe: PoolType::LifinityPoolType,
    //     dir_path: "../pools/lifinity/".to_string(),
    // };
    // pool_dirs.push(saber_dir);

    let mut token_mints = fetch_tokens(); // Fetch token mints dynamically
    let mut pools = vec![];

    let mut update_pks = vec![];
    let mut update_pks_lengths = vec![];
    let mut all_mint_idxs = vec![];

    let mut mint2idx = HashMap::new();
    let mut graph_edges = vec![];

    info!("Extracting pool + mints...");
    for pool_dir in pool_dirs {
        debug!("Pool dir: {:#?}", pool_dir);
        let pool_paths = read_json_dir(&pool_dir.dir_path);

        for pool_path in pool_paths {
            let json_str = std::fs::read_to_string(&pool_path).unwrap();
            let pool = pool_factory(&pool_dir.tipe, &json_str);

            let pool_mints = pool.get_mints();
            if pool_mints.len() != 2 {
                warn!("Skipping pool with mints != 2: {:?}", pool_path);
                continue;
            }

            let mut mint_idxs = vec![];
            for mint in pool_mints {
                let idx;
                if !token_mints.contains(&mint) {
                    idx = token_mints.len();
                    mint2idx.insert(mint, idx);
                    token_mints.push(mint);
                    graph_edges.push(HashSet::new());
                } else {
                    idx = *mint2idx.get(&mint).unwrap();
                }
                mint_idxs.push(idx);
            }

            let update_accounts = pool.get_update_accounts();
            update_pks_lengths.push(update_accounts.len());
            update_pks.push(update_accounts);

            let mint0_idx = mint_idxs[0];
            let mint1_idx = mint_idxs[1];

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

    info!("Added {:?} mints", token_mints.len());
    info!("Added {:?} pools", pools.len());

    let usdc_mint = Pubkey::from_str("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v").unwrap();
    let start_mint = usdc_mint;
    let start_mint_idx = *mint2idx.get(&start_mint).unwrap();

    let owner: &Keypair = rc_owner.borrow();
    let owner_start_addr = derive_token_address(&owner.pubkey(), &start_mint);
    update_pks.push(owner_start_addr);

    info!("Getting pool amounts...");
    let mut update_accounts = vec![];
    for token_addr_chunk in update_pks.chunks(99) {
        let accounts = connection.get_multiple_accounts(token_addr_chunk).unwrap_or_else(|e| {
            warn!("Failed to get accounts: {}", e);
            vec![None; token_addr_chunk.len()]
        });
        update_accounts.push(accounts);
    }
    let mut update_accounts = update_accounts
        .concat()
        .into_iter()
        .filter(|s| s.is_some())
        .collect::<Vec<Option<Account>>>();

    info!("Update accounts count: {:?}", update_accounts.len());
    let init_token_acc = update_accounts.pop().unwrap().unwrap();
    let init_token_balance = unpack_token_account(&init_token_acc.data).amount as u128;
    info!(
        "Init token acc: {:?}, balance: {:#}",
        init_token_acc, init_token_balance
    );
    info!("Starting balance = {}", init_token_balance);

    info!("Setting up exchange graph...");
    let mut graph = PoolGraph::new();
    let mut pool_count = 0;
    let mut account_ptr = 0;

    for pool in pools.into_iter() {
        let length = update_pks_lengths[pool_count];
        let _account_slice = &update_accounts[account_ptr..account_ptr + length].to_vec();
        account_ptr += length;

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

    info!("Searching for arbitrages...");
    let min_swap_amount = 10_u128.pow(6_u32); // scaled! -- 1 USDC
    let mut swap_start_amount = init_token_balance; // scaled!
    let mut sent_arbs = HashSet::new();

    for _ in 0..4 {
        let fees = calculate_fees(swap_start_amount, config.fee_percentage);
        let net_amount = swap_start_amount - fees;

        arbitrager.brute_force_search(
            start_mint_idx,
            net_amount,
            swap_start_amount,
            vec![start_mint_idx],
            vec![],
            &mut sent_arbs,
        );

        swap_start_amount /= 2; // half input amount and search again
        if swap_start_amount < min_swap_amount {
            break;
        }
    }
}
