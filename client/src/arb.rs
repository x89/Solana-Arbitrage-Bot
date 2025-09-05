use anchor_client::solana_client::rpc_client::RpcClient;
use anchor_client::solana_client::rpc_config::RpcSendTransactionConfig;

use anchor_client::solana_sdk::pubkey::Pubkey;

use anchor_client::solana_sdk::signature::{Keypair, Signer};
use anchor_client::{Cluster, Program};
use std::collections::{HashMap, HashSet};

use solana_sdk::instruction::Instruction;
use solana_sdk::transaction::Transaction;

use std::borrow::Borrow;
use std::rc::Rc;

use std::vec;

use log::info;

use tmp::accounts as tmp_accounts;
use tmp::instruction as tmp_ix;

use crate::pool::PoolOperations;

use crate::utils::{derive_token_address, PoolGraph, PoolIndex, PoolQuote};

pub struct Arbitrager {
    pub token_mints: Vec<Pubkey>,
    pub graph_edges: Vec<HashSet<usize>>, // used for quick searching over the graph
    pub graph: PoolGraph,
    pub cluster: Cluster,
    // vv -- need to clone these explicitly -- vv
    pub owner: Rc<Keypair>,
    pub program: Program,
    pub connection: RpcClient,
}

impl Arbitrager {
    pub fn brute_force_search(
        &self,
        start_mint_idx: usize,
        init_balance: u128,
        curr_balance: u128,
        path: Vec<usize>,
        pool_path: Vec<PoolQuote>,
        sent_arbs: &mut HashSet<String>,
    ) {
        let src_curr = path[path.len() - 1]; // last mint
        let src_mint = self.token_mints[src_curr];

        let out_edges = &self.graph_edges[src_curr];

        // path = 4 = A -> B -> C -> D
        // path >= 5 == not valid bc max tx size is swaps
        if path.len() == 4 {
            return;
        };

        for dst_mint_idx in out_edges {
            let pools = self
                .graph
                .0
                .get(&PoolIndex(src_curr))
                .unwrap()
                .0
                .get(&PoolIndex(*dst_mint_idx))
                .unwrap();

            if path.contains(dst_mint_idx) && *dst_mint_idx != start_mint_idx {
                continue;
            }

            let dst_mint_idx = *dst_mint_idx;
            let dst_mint = self.token_mints[dst_mint_idx];

            for pool in pools {
                let new_balance =
                    pool.0
                        .get_quote_with_amounts_scaled(curr_balance, &src_mint, &dst_mint);

                let mut new_path = path.clone();
                new_path.push(dst_mint_idx);

                let mut new_pool_path = pool_path.clone();
                new_pool_path.push(pool.clone()); // clone the pointer

                if dst_mint_idx == start_mint_idx {
                    // info!("{:?} -> {:?} (-{:?})", init_balance, new_balance, init_balance - new_balance);

                    // if new_balance > init_balance - 1086310399 {
                    if new_balance > init_balance {
                        // ... profitable arb!
                        info!("found arbitrage: {:?} -> {:?}", init_balance, new_balance);

                        // check if arb was sent with a larger size
                        // key = {mint_path}{pool_names}
                        let mint_keys: Vec<String> =
                            new_path.clone().iter_mut().map(|i| i.to_string()).collect();
                        let pool_keys: Vec<String> =
                            new_pool_path.iter().map(|p| p.0.get_name()).collect();
                        let arb_key = format!("{}{}", mint_keys.join(""), pool_keys.join(""));
                        if sent_arbs.contains(&arb_key) {
                            info!("arb already sent...");
                            continue; // dont re-send an already sent arb -- bad for network
                        } else {
                            sent_arbs.insert(arb_key);
                        }

                        let ixs = self.get_arbitrage_instructions(
                            init_balance,
                            &new_path,
                            &new_pool_path,
                        );
                        self.send_ixs(ixs);
                    }
                } else if !path.contains(&dst_mint_idx) {
                    // ... search deeper
                    self.brute_force_search(
                        start_mint_idx,
                        init_balance,
                        new_balance,   // !
                        new_path,      // !
                        new_pool_path, // !
                        sent_arbs,
                    );
                }
            }
        }
    }

    fn get_arbitrage_instructions(
        &self,
        swap_start_amount: u128,
        mint_idxs: &Vec<usize>,
        pools: &Vec<PoolQuote>,
    ) -> Vec<Instruction> {
        // gather swap ixs
        let mut ixs = vec![];
        let (swap_state_pda, _) =
            Pubkey::find_program_address(&[b"swap_state"], &self.program.id());

        let src_mint = self.token_mints[mint_idxs[0]];
        let src_ata = derive_token_address(&self.owner.pubkey(), &src_mint);

        // initialize swap ix
        let ix = self
            .program
            .request()
            .accounts(tmp_accounts::TokenAndSwapState {
                src: src_ata,
                swap_state: swap_state_pda,
            })
            .args(tmp_ix::StartSwap {
                swap_input: swap_start_amount as u64,
            })
            .instructions()
            .unwrap();
        ixs.push(ix);

        for i in 0..mint_idxs.len() - 1 {
            let [mint_idx0, mint_idx1] = [mint_idxs[i], mint_idxs[i + 1]];
            let [mint0, mint1] = [self.token_mints[mint_idx0], self.token_mints[mint_idx1]];
            let pool = &pools[i];

            let swap_ix = pool
                .0
                .swap_ix(&self.program, &self.owner.pubkey(), &mint0, &mint1);
            ixs.push(swap_ix);
        }

        // PROFIT OR REVERT instruction
        let ix = self
            .program
            .request()
            .accounts(tmp_accounts::TokenAndSwapState {
                src: src_ata,
                swap_state: swap_state_pda,
            })
            .args(tmp_ix::ProfitOrRevert {})
            .instructions()
            .unwrap();
        ixs.push(ix);

        // flatten to Vec<Instructions>
        ixs.concat()
    }

    fn send_ixs(&self, ixs: Vec<Instruction>) {
        let owner: &Keypair = self.owner.borrow();
        let tx = Transaction::new_signed_with_payer(
            &ixs,
            Some(&owner.pubkey()),
            &[owner],
            self.connection.get_latest_blockhash().unwrap(),
        );

        if self.cluster == Cluster::Localnet {
            let res = self.connection.simulate_transaction(&tx).unwrap();
            println!("{:#?}", res);
        } else if self.cluster == Cluster::Mainnet {
            let signature = self
                .connection
                .send_transaction_with_config(
                    &tx,
                    RpcSendTransactionConfig {
                        skip_preflight: true,
                        ..RpcSendTransactionConfig::default()
                    },
                )
                .unwrap();
            println!("signature: {:?}", signature);
        }
    }
}
