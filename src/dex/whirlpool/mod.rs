pub mod constants;
pub mod state;

use crate::dex::whirlpool::state::{Whirlpool, TICK_ARRAY_SIZE};
use crate::dex::whirlpool::constants::{MAX_TICK_INDEX, MIN_TICK_INDEX};
use solana_program::instruction::AccountMeta;
use solana_program::pubkey::Pubkey;

pub type TickArrayStartIndexes = (i32, Option<i32>, Option<i32>);

pub fn derive_start_tick(curr_tick: i32, tick_spacing: u16) -> i32 {
    let num_of_ticks_in_array = TICK_ARRAY_SIZE as i32 * tick_spacing as i32;
    let rem = curr_tick % num_of_ticks_in_array;
    if curr_tick < 0 && rem != 0 {
        curr_tick - rem - num_of_ticks_in_array
    } else {
        curr_tick - rem
    }
}

pub fn derive_first_tick_array_start_tick(curr_tick: i32, tick_spacing: u16, shifted: bool) -> i32 {
    let tick = if shifted {
        curr_tick + tick_spacing as i32
    } else {
        curr_tick
    };
    derive_start_tick(tick, tick_spacing)
}

pub fn derive_tick_array_start_indexes(
    curr_tick: i32,
    tick_spacing: u16,
    a_to_b: bool,
) -> TickArrayStartIndexes {
    let ta0_start_index = derive_first_tick_array_start_tick(curr_tick, tick_spacing, !a_to_b);
    let ta1_start_index_opt = derive_next_start_tick_in_seq(ta0_start_index, tick_spacing, a_to_b);
    let ta2_start_index_opt = ta1_start_index_opt
        .and_then(|nsi| derive_next_start_tick_in_seq(nsi, tick_spacing, a_to_b));
    (ta0_start_index, ta1_start_index_opt, ta2_start_index_opt)
}

pub fn derive_next_start_tick_in_seq(
    start_tick: i32,
    tick_spacing: u16,
    a_to_b: bool,
) -> Option<i32> {
    let num_of_ticks_in_array = TICK_ARRAY_SIZE as i32 * tick_spacing as i32;
    let potential_last = if a_to_b {
        start_tick - num_of_ticks_in_array
    } else {
        start_tick + num_of_ticks_in_array
    };
    if potential_last < MAX_TICK_INDEX && potential_last > MIN_TICK_INDEX {
        Some(potential_last)
    } else {
        None
    }
}

pub fn get_tick_array_address(
    whirlpool: &Pubkey,
    start_tick_index: i32,
    program_id: &Pubkey,
) -> Pubkey {
    let start_tick_index_str = start_tick_index.to_string();
    let seeds = &[
        b"tick_array",
        whirlpool.as_ref(),
        start_tick_index_str.as_bytes(),
    ];

    Pubkey::find_program_address(seeds, program_id).0
}

pub fn update_tick_array_accounts_for_onchain(
    whirlpool: &Whirlpool,
    whirlpool_pk: &Pubkey,
    whirlpool_program_id: &Pubkey,
) -> Vec<AccountMeta> {
    let tick_array_starts =
        derive_tick_array_start_indexes(whirlpool.tick_current_index, whirlpool.tick_spacing, true);
    let tick_array_reverse_starts = derive_tick_array_start_indexes(
        whirlpool.tick_current_index,
        whirlpool.tick_spacing,
        false,
    );

    let tick_array_pks = vec![
        AccountMeta::new(
            get_tick_array_address(
                whirlpool_pk,
                tick_array_reverse_starts
                    .1
                    .unwrap_or(tick_array_reverse_starts.0),
                whirlpool_program_id,
            ),
            false,
        ),
        AccountMeta::new(
            get_tick_array_address(whirlpool_pk, tick_array_starts.0, whirlpool_program_id),
            false,
        ),
        AccountMeta::new(
            get_tick_array_address(
                whirlpool_pk,
                tick_array_starts.1.unwrap_or(tick_array_starts.0),
                whirlpool_program_id,
            ),
            false,
        ),
    ];
    tick_array_pks
}
