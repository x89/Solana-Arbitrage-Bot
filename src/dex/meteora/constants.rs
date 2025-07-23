use solana_program::pubkey::Pubkey;
use std::str::FromStr;

pub fn dlmm_program_id() -> Pubkey {
    Pubkey::from_str("LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo").unwrap()
}

pub fn dlmm_event_authority() -> Pubkey {
    Pubkey::from_str("D1ZN9Wj1fRSUQfCjhvnu1hqDMT7hzjzBBpi12nVniYD6").unwrap()
}

pub fn damm_program_id() -> Pubkey {
    Pubkey::from_str("Eo7WjKq67rjJQSZxS6z3YkapzY3eMj6Xy8X5EQVn5UaB").unwrap()
}

pub fn vault_program_id() -> Pubkey {
    Pubkey::from_str("24Uqj9JCLxUeoC3hGfh5W3s9FM9uCHDS2SG3LYwBpyTi").unwrap()
}

pub fn damm_v2_program_id() -> Pubkey {
    Pubkey::from_str("cpamdpZCGKUy5JxQXB4dcpGPiikHawvSWAd6mEn1sGG").unwrap()
}

pub fn damm_v2_event_authority() -> Pubkey {
    Pubkey::from_str("3rmHSu74h1ZcmAisVcWerTCiRDQbUrBKmcwptYGjHfet").unwrap()
}

pub fn damm_v2_pool_authority() -> Pubkey {
    Pubkey::from_str("HLnpSz9h2S4hiLQ43rnSD9XkcUThA7B8hQMKmDaiTLcC").unwrap()
}

pub const BIN_ARRAY: &[u8] = b"bin_array";
