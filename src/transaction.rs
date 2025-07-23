use crate::config::Config;
use crate::dex::raydium::{raydium_authority, raydium_cp_authority};
use crate::dex::solfi::constants::solfi_program_id;
use crate::dex::vertigo::constants::vertigo_program_id;
use crate::pools::MintPoolData;
use solana_client::rpc_client::RpcClient;
use solana_program::instruction::Instruction;
use solana_sdk::address_lookup_table::AddressLookupTableAccount;
use solana_sdk::commitment_config::CommitmentLevel;
use solana_sdk::compute_budget::ComputeBudgetInstruction;
use solana_sdk::hash::Hash;
use solana_sdk::message::v0::Message;
use solana_sdk::signature::{Keypair, Signature};
use solana_sdk::signer::Signer;
use solana_sdk::transaction::VersionedTransaction;
use std::sync::Arc;
use tracing::{debug, error, info};

use crate::constants::sol_mint;
use crate::dex::meteora::constants::{
    damm_program_id, damm_v2_event_authority, damm_v2_pool_authority, damm_v2_program_id,
    dlmm_event_authority, dlmm_program_id, vault_program_id,
};
use crate::dex::pump::constants::{pump_fee_wallet, pump_program_id};
use crate::dex::raydium::constants::{
    raydium_clmm_program_id, raydium_cp_program_id, raydium_program_id,
};
use crate::dex::whirlpool::constants::whirlpool_program_id;
use solana_program::instruction::AccountMeta;
use solana_program::pubkey::Pubkey;
use solana_program::system_program;
use spl_associated_token_account::ID as associated_token_program_id;
use spl_token::ID as token_program_id;
use std::str::FromStr;

pub async fn build_and_send_transaction(
    wallet_kp: &Keypair,
    config: &Config,
    mint_pool_data: &MintPoolData,
    rpc_clients: &[Arc<RpcClient>],
    blockhash: Hash,
    address_lookup_table_accounts: &[AddressLookupTableAccount],
) -> anyhow::Result<Vec<Signature>> {
    let enable_flashloan = config.flashloan.as_ref().map_or(false, |k| k.enabled);
    let compute_unit_limit = config.bot.compute_unit_limit;
    let mut instructions = vec![];
    // Add a random number here to make each transaction unique
    let compute_budget_ix = ComputeBudgetInstruction::set_compute_unit_limit(
        compute_unit_limit + rand::random::<u32>() % 1000,
    );
    instructions.push(compute_budget_ix);

    let compute_unit_price = config.spam.as_ref().map_or(1000, |s| s.compute_unit_price);
    let compute_budget_price_ix =
        ComputeBudgetInstruction::set_compute_unit_price(compute_unit_price);
    instructions.push(compute_budget_price_ix);

    let swap_ix = create_swap_instruction(
        wallet_kp,
        mint_pool_data,
        compute_unit_limit as u64,
        enable_flashloan,
    )?;

    let mut all_instructions = instructions.clone();

    debug!("Adding swap instruction");
    all_instructions.push(swap_ix);

    let message = Message::try_compile(
        &wallet_kp.pubkey(),
        &all_instructions,
        address_lookup_table_accounts,
        blockhash,
    )?;

    let tx = VersionedTransaction::try_new(
        solana_sdk::message::VersionedMessage::V0(message),
        &[wallet_kp],
    )?;

    let max_retries = config
        .spam
        .as_ref()
        .and_then(|s| s.max_retries)
        .unwrap_or(3);

    let mut signatures = Vec::new();

    for (i, client) in rpc_clients.iter().enumerate() {
        debug!("Sending transaction through RPC client {}", i);

        let signature = match send_transaction_with_retries(client, &tx, max_retries).await {
            Ok(sig) => sig,
            Err(e) => {
                error!("Failed to send transaction through RPC client {}: {}", i, e);
                continue;
            }
        };

        info!(
            "Transaction sent successfully through RPC client {}: {}",
            i, signature
        );
        signatures.push(signature);
    }

    Ok(signatures)
}

async fn send_transaction_with_retries(
    client: &RpcClient,
    tx: &VersionedTransaction,
    max_retries: u64,
) -> anyhow::Result<Signature> {
    Ok(client.send_transaction_with_config(
        tx,
        solana_client::rpc_config::RpcSendTransactionConfig {
            skip_preflight: true,
            max_retries: Some(max_retries as usize),
            preflight_commitment: Some(CommitmentLevel::Confirmed),
            ..Default::default()
        },
    )?)
}

/// Helper function to derive the vault token account PDA address for a given mint
pub fn derive_vault_token_account(program_id: &Pubkey, mint: &Pubkey) -> (Pubkey, u8) {
    Pubkey::find_program_address(&[b"vault_token_account", mint.as_ref()], program_id)
}

// See https://docs.solanamevbot.com/home/onchain-bot/onchain-program for more information
fn create_swap_instruction(
    wallet_kp: &Keypair,
    mint_pool_data: &MintPoolData,
    compute_unit_limit: u64,
    use_flashloan: bool,
) -> anyhow::Result<Instruction> {
    debug!("Creating swap instruction for all DEX types");

    let executor_program_id =
        Pubkey::from_str("MEViEnscUm6tsQRoGd9h6nLQaQspKj7DB2M5FwM3Xvz").unwrap();
    let fee_collector = Pubkey::from_str("6AGB9kqgSp2mQXwYpdrV4QVV8urvCaDS35U1wsLssy6H").unwrap();

    let pump_global_config =
        Pubkey::from_str("ADyA8hdefvWN2dbGGWFotbzWxrAvLW83WG6QCVXvJKqw").unwrap();
    let pump_authority = Pubkey::from_str("GS4CU59F31iL7aR2Q8zVS8DRrcRnXX1yjQ66TqNVQnaR").unwrap();
    let sysvar_instructions =
        Pubkey::from_str("Sysvar1nstructions1111111111111111111111111").unwrap();

    let wallet = wallet_kp.pubkey();
    let sol_mint_pubkey = sol_mint();
    let wallet_sol_account = mint_pool_data.wallet_wsol_account;

    let mut accounts = vec![
        AccountMeta::new_readonly(wallet, true), // 0. Wallet (signer)
        AccountMeta::new_readonly(sol_mint_pubkey, false), // 1. SOL mint
        AccountMeta::new(fee_collector, false),  // 2. Fee collector
        AccountMeta::new(wallet_sol_account, false), // 3. Wallet SOL account
        AccountMeta::new_readonly(token_program_id, false), // 4. Token program
        AccountMeta::new_readonly(system_program::ID, false), // 5. System program
        AccountMeta::new_readonly(associated_token_program_id, false), // 6. Associated Token program
    ];

    let base_mint = sol_mint_pubkey;

    if use_flashloan {
        accounts.push(AccountMeta::new_readonly(
            Pubkey::from_str("5LFpzqgsxrSfhKwbaFiAEJ2kbc9QyimjKueswsyU4T3o").unwrap(),
            false,
        ));
        let token_pda = derive_vault_token_account(
            &Pubkey::from_str("MEViEnscUm6tsQRoGd9h6nLQaQspKj7DB2M5FwM3Xvz").unwrap(),
            &base_mint,
        );
        accounts.push(AccountMeta::new(token_pda.0, false));
    }

    accounts.push(AccountMeta::new_readonly(mint_pool_data.mint, false));
    accounts.push(AccountMeta::new_readonly(
        mint_pool_data.token_program,
        false,
    )); // Token program (SPL Token or Token 2022)
    let wallet_x_account =
        spl_associated_token_account::get_associated_token_address_with_program_id(
            &wallet,
            &mint_pool_data.mint,
            &mint_pool_data.token_program,
        );
    accounts.push(AccountMeta::new(wallet_x_account, false));

    for pool in &mint_pool_data.raydium_pools {
        accounts.push(AccountMeta::new_readonly(raydium_program_id(), false));
        accounts.push(AccountMeta::new_readonly(raydium_authority(), false)); // Raydium authority
        accounts.push(AccountMeta::new(pool.pool, false));
        accounts.push(AccountMeta::new(pool.token_vault, false));
        accounts.push(AccountMeta::new(pool.sol_vault, false));
    }

    for pool in &mint_pool_data.raydium_cp_pools {
        accounts.push(AccountMeta::new_readonly(raydium_cp_program_id(), false));
        accounts.push(AccountMeta::new_readonly(raydium_cp_authority(), false)); // Raydium CP authority
        accounts.push(AccountMeta::new(pool.pool, false));
        accounts.push(AccountMeta::new_readonly(pool.amm_config, false));
        accounts.push(AccountMeta::new(pool.token_vault, false));
        accounts.push(AccountMeta::new(pool.sol_vault, false));
        accounts.push(AccountMeta::new(pool.observation, false));
    }

    for pool in &mint_pool_data.pump_pools {
        accounts.push(AccountMeta::new_readonly(pump_program_id(), false));
        accounts.push(AccountMeta::new_readonly(pump_global_config, false));
        accounts.push(AccountMeta::new_readonly(pump_authority, false));
        accounts.push(AccountMeta::new_readonly(pump_fee_wallet(), false));
        accounts.push(AccountMeta::new_readonly(pool.pool, false));
        accounts.push(AccountMeta::new(pool.token_vault, false));
        accounts.push(AccountMeta::new(pool.sol_vault, false));
        accounts.push(AccountMeta::new(pool.fee_token_wallet, false));
        accounts.push(AccountMeta::new(pool.coin_creator_vault_ata, false));
        accounts.push(AccountMeta::new_readonly(
            pool.coin_creator_vault_authority,
            false,
        ));
    }

    for pair in &mint_pool_data.dlmm_pairs {
        accounts.push(AccountMeta::new_readonly(dlmm_program_id(), false));
        accounts.push(AccountMeta::new(dlmm_event_authority(), false)); // DLMM event authority
        if let Some(memo_program) = pair.memo_program {
            accounts.push(AccountMeta::new_readonly(memo_program, false)); // Token 2022 memo program
        }
        accounts.push(AccountMeta::new(pair.pair, false));
        accounts.push(AccountMeta::new(pair.token_vault, false));
        accounts.push(AccountMeta::new(pair.sol_vault, false));
        accounts.push(AccountMeta::new(pair.oracle, false));
        for bin_array in &pair.bin_arrays {
            accounts.push(AccountMeta::new(*bin_array, false));
        }
    }

    for pool in &mint_pool_data.whirlpool_pools {
        accounts.push(AccountMeta::new_readonly(whirlpool_program_id(), false));
        if let Some(memo_program) = pool.memo_program {
            accounts.push(AccountMeta::new_readonly(memo_program, false)); // Token 2022 memo program
        }
        accounts.push(AccountMeta::new(pool.pool, false));
        accounts.push(AccountMeta::new(pool.oracle, false));
        accounts.push(AccountMeta::new(pool.x_vault, false));
        accounts.push(AccountMeta::new(pool.y_vault, false));
        for tick_array in &pool.tick_arrays {
            accounts.push(AccountMeta::new(*tick_array, false));
        }
    }

    for pool in &mint_pool_data.raydium_clmm_pools {
        accounts.push(AccountMeta::new_readonly(raydium_clmm_program_id(), false));
        if let Some(memo_program) = pool.memo_program {
            accounts.push(AccountMeta::new_readonly(memo_program, false)); // Token 2022 memo program
        }
        accounts.push(AccountMeta::new(pool.pool, false));
        accounts.push(AccountMeta::new_readonly(pool.amm_config, false));
        accounts.push(AccountMeta::new(pool.observation_state, false));
        accounts.push(AccountMeta::new(pool.bitmap_extension, false));
        accounts.push(AccountMeta::new(pool.x_vault, false));
        accounts.push(AccountMeta::new(pool.y_vault, false));
        for tick_array in &pool.tick_arrays {
            accounts.push(AccountMeta::new(*tick_array, false));
        }
    }

    for pool in &mint_pool_data.meteora_damm_pools {
        accounts.push(AccountMeta::new_readonly(damm_program_id(), false));
        accounts.push(AccountMeta::new_readonly(vault_program_id(), false));
        accounts.push(AccountMeta::new(pool.pool, false));
        accounts.push(AccountMeta::new(pool.token_x_vault, false));
        accounts.push(AccountMeta::new(pool.token_sol_vault, false));
        accounts.push(AccountMeta::new(pool.token_x_token_vault, false));
        accounts.push(AccountMeta::new(pool.token_sol_token_vault, false));
        accounts.push(AccountMeta::new(pool.token_x_lp_mint, false));
        accounts.push(AccountMeta::new(pool.token_sol_lp_mint, false));
        accounts.push(AccountMeta::new(pool.token_x_pool_lp, false));
        accounts.push(AccountMeta::new(pool.token_sol_pool_lp, false));
        accounts.push(AccountMeta::new(pool.admin_token_fee_x, false));
        accounts.push(AccountMeta::new(pool.admin_token_fee_sol, false));
    }

    for pool in &mint_pool_data.meteora_damm_v2_pools {
        accounts.push(AccountMeta::new_readonly(damm_v2_program_id(), false));
        accounts.push(AccountMeta::new_readonly(damm_v2_event_authority(), false));
        accounts.push(AccountMeta::new_readonly(damm_v2_pool_authority(), false));
        accounts.push(AccountMeta::new(pool.pool, false));
        accounts.push(AccountMeta::new(pool.token_x_vault, false));
        accounts.push(AccountMeta::new(pool.token_sol_vault, false));
    }

    for pool in &mint_pool_data.solfi_pools {
        accounts.push(AccountMeta::new_readonly(solfi_program_id(), false));
        accounts.push(AccountMeta::new_readonly(sysvar_instructions, false));
        accounts.push(AccountMeta::new(pool.pool, false));
        accounts.push(AccountMeta::new(pool.token_x_vault, false));
        accounts.push(AccountMeta::new(pool.token_sol_vault, false));
    }

    for pool in &mint_pool_data.vertigo_pools {
        accounts.push(AccountMeta::new_readonly(vertigo_program_id(), false));
        accounts.push(AccountMeta::new(pool.pool, false));
        accounts.push(AccountMeta::new_readonly(pool.pool_owner, false));
        accounts.push(AccountMeta::new(pool.token_x_vault, false));
        accounts.push(AccountMeta::new(pool.token_sol_vault, false));
    }

    let mut data = vec![26u8];

    let minimum_profit: u64 = 0;
    // When true, the bot will not fail the transaction even when it can't find a profitable arbitrage. It will just do nothing and succeed.
    let no_failure_mode = false;

    data.extend_from_slice(&minimum_profit.to_le_bytes());
    data.extend_from_slice(&compute_unit_limit.to_le_bytes());
    data.extend_from_slice(if no_failure_mode { &[1] } else { &[0] });
    data.extend_from_slice(&0u16.to_le_bytes()); // Keep this 0.
    data.extend_from_slice(if use_flashloan { &[1] } else { &[0] });

    Ok(Instruction {
        program_id: executor_program_id,
        accounts,
        data,
    })
}
