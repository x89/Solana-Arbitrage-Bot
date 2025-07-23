use solana_program::pubkey::Pubkey;

pub const NUM_REWARDS: usize = 3;

#[derive(Clone, Copy, Debug)]
pub struct Whirlpool {
    pub whirlpools_config: Pubkey, // 32
    pub whirlpool_bump: [u8; 1],   // 1

    pub tick_spacing: u16,          // 2
    pub tick_spacing_seed: [u8; 2], // 2

    pub fee_rate: u16, // 2

    pub protocol_fee_rate: u16, // 2

    pub liquidity: u128, // 16

    pub sqrt_price: u128,        // 16
    pub tick_current_index: i32, // 4

    pub protocol_fee_owed_a: u64, // 8
    pub protocol_fee_owed_b: u64, // 8

    pub token_mint_a: Pubkey,  // 32
    pub token_vault_a: Pubkey, // 32

    pub fee_growth_global_a: u128, // 16

    pub token_mint_b: Pubkey,  // 32
    pub token_vault_b: Pubkey, // 32

    pub fee_growth_global_b: u128, // 16

    pub reward_last_updated_timestamp: u64, // 8

    pub reward_infos: [WhirlpoolRewardInfo; NUM_REWARDS], // 384
}

impl Whirlpool {
    pub const LEN: usize = 8 + 261 + 384;
}

#[derive(Copy, Clone, Debug)]
pub struct WhirlpoolRewardInfo {
    pub mint: Pubkey,
    pub vault: Pubkey,
    pub authority: Pubkey,
    pub emissions_per_second_x64: u128,
    pub growth_global_x64: u128,
}

#[derive(Clone, Debug)]
pub struct TickArray {
    pub start_tick_index: i32,
    pub ticks: [Tick; TICK_ARRAY_SIZE],
    pub whirlpool: Pubkey,
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Tick {
    pub initialized: bool,
    pub liquidity_net: i128,
    pub liquidity_gross: u128,
    pub fee_growth_outside_a: u128,
    pub fee_growth_outside_b: u128,
    pub reward_growths_outside: [u128; NUM_REWARDS],
}

impl Tick {
    pub fn check_is_valid_start_tick(tick_index: i32, tick_spacing: u16) -> bool {
        tick_index % (tick_spacing as i32 * TICK_ARRAY_SIZE as i32) == 0
    }
}

pub const TICK_ARRAY_SIZE: usize = 88;

use std::io::{Error, ErrorKind, Result};

impl Whirlpool {
    pub fn try_deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < Self::LEN {
            return Err(Error::new(ErrorKind::InvalidData, "data too short for Whirlpool"));
        }

        let data = &data[8..];
        
        let mut offset = 0;
        
        let mut whirlpools_config = [0u8; 32];
        whirlpools_config.copy_from_slice(&data[offset..offset+32]);
        let whirlpools_config = Pubkey::new_from_array(whirlpools_config);
        offset += 32;
        
        let mut whirlpool_bump = [0u8; 1];
        whirlpool_bump.copy_from_slice(&data[offset..offset+1]);
        offset += 1;
        
        let tick_spacing = u16::from_le_bytes([data[offset], data[offset+1]]);
        offset += 2;
        
        let mut tick_spacing_seed = [0u8; 2];
        tick_spacing_seed.copy_from_slice(&data[offset..offset+2]);
        offset += 2;
        
        let fee_rate = u16::from_le_bytes([data[offset], data[offset+1]]);
        offset += 2;
        
        let protocol_fee_rate = u16::from_le_bytes([data[offset], data[offset+1]]);
        offset += 2;
        
        let mut liquidity_bytes = [0u8; 16];
        liquidity_bytes.copy_from_slice(&data[offset..offset+16]);
        let liquidity = u128::from_le_bytes(liquidity_bytes);
        offset += 16;
        
        let mut sqrt_price_bytes = [0u8; 16];
        sqrt_price_bytes.copy_from_slice(&data[offset..offset+16]);
        let sqrt_price = u128::from_le_bytes(sqrt_price_bytes);
        offset += 16;
        
        let mut tick_current_index_bytes = [0u8; 4];
        tick_current_index_bytes.copy_from_slice(&data[offset..offset+4]);
        let tick_current_index = i32::from_le_bytes(tick_current_index_bytes);
        offset += 4;
        
        let mut protocol_fee_owed_a_bytes = [0u8; 8];
        protocol_fee_owed_a_bytes.copy_from_slice(&data[offset..offset+8]);
        let protocol_fee_owed_a = u64::from_le_bytes(protocol_fee_owed_a_bytes);
        offset += 8;
        
        let mut protocol_fee_owed_b_bytes = [0u8; 8];
        protocol_fee_owed_b_bytes.copy_from_slice(&data[offset..offset+8]);
        let protocol_fee_owed_b = u64::from_le_bytes(protocol_fee_owed_b_bytes);
        offset += 8;
        
        let mut token_mint_a_bytes = [0u8; 32];
        token_mint_a_bytes.copy_from_slice(&data[offset..offset+32]);
        let token_mint_a = Pubkey::new_from_array(token_mint_a_bytes);
        offset += 32;
        
        let mut token_vault_a_bytes = [0u8; 32];
        token_vault_a_bytes.copy_from_slice(&data[offset..offset+32]);
        let token_vault_a = Pubkey::new_from_array(token_vault_a_bytes);
        offset += 32;
        
        let mut fee_growth_global_a_bytes = [0u8; 16];
        fee_growth_global_a_bytes.copy_from_slice(&data[offset..offset+16]);
        let fee_growth_global_a = u128::from_le_bytes(fee_growth_global_a_bytes);
        offset += 16;
        
        let mut token_mint_b_bytes = [0u8; 32];
        token_mint_b_bytes.copy_from_slice(&data[offset..offset+32]);
        let token_mint_b = Pubkey::new_from_array(token_mint_b_bytes);
        offset += 32;
        
        let mut token_vault_b_bytes = [0u8; 32];
        token_vault_b_bytes.copy_from_slice(&data[offset..offset+32]);
        let token_vault_b = Pubkey::new_from_array(token_vault_b_bytes);
        offset += 32;
        
        let mut fee_growth_global_b_bytes = [0u8; 16];
        fee_growth_global_b_bytes.copy_from_slice(&data[offset..offset+16]);
        let fee_growth_global_b = u128::from_le_bytes(fee_growth_global_b_bytes);
        offset += 16;
        
        let mut reward_last_updated_timestamp_bytes = [0u8; 8];
        reward_last_updated_timestamp_bytes.copy_from_slice(&data[offset..offset+8]);
        let reward_last_updated_timestamp = u64::from_le_bytes(reward_last_updated_timestamp_bytes);
        offset += 8;
        
        let mut reward_infos = [WhirlpoolRewardInfo {
            mint: Pubkey::default(),
            vault: Pubkey::default(),
            authority: Pubkey::default(),
            emissions_per_second_x64: 0,
            growth_global_x64: 0,
        }; NUM_REWARDS];
        
        for i in 0..NUM_REWARDS {
            let mut mint_bytes = [0u8; 32];
            mint_bytes.copy_from_slice(&data[offset..offset+32]);
            reward_infos[i].mint = Pubkey::new_from_array(mint_bytes);
            offset += 32;
            
            let mut vault_bytes = [0u8; 32];
            vault_bytes.copy_from_slice(&data[offset..offset+32]);
            reward_infos[i].vault = Pubkey::new_from_array(vault_bytes);
            offset += 32;
            
            let mut authority_bytes = [0u8; 32];
            authority_bytes.copy_from_slice(&data[offset..offset+32]);
            reward_infos[i].authority = Pubkey::new_from_array(authority_bytes);
            offset += 32;
            
            let mut emissions_bytes = [0u8; 16];
            emissions_bytes.copy_from_slice(&data[offset..offset+16]);
            reward_infos[i].emissions_per_second_x64 = u128::from_le_bytes(emissions_bytes);
            offset += 16;
            
            let mut growth_bytes = [0u8; 16];
            growth_bytes.copy_from_slice(&data[offset..offset+16]);
            reward_infos[i].growth_global_x64 = u128::from_le_bytes(growth_bytes);
            offset += 16;
        }
        
        Ok(Whirlpool {
            whirlpools_config,
            whirlpool_bump,
            tick_spacing,
            tick_spacing_seed,
            fee_rate,
            protocol_fee_rate,
            liquidity,
            sqrt_price,
            tick_current_index,
            protocol_fee_owed_a,
            protocol_fee_owed_b,
            token_mint_a,
            token_vault_a,
            fee_growth_global_a,
            token_mint_b,
            token_vault_b,
            fee_growth_global_b,
            reward_last_updated_timestamp,
            reward_infos,
        })
    }
}
