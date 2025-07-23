use anchor_gen::generate_cpi_crate;
use hexlit::hex;

generate_cpi_crate!("./idl.json");
declare_id!("Eo7WjKq67rjJQSZxS6z3YkapzY3eMj6Xy8X5EQVn5UaB");

pub fn encode_swap(amount_in: u64, min_amount_out: u64) -> Vec<u8> {
    let mut data = Vec::default();
    // NB: SHA256("global:swap").
    data.extend(hex!("f8c69e91e17587c8"));
    instruction::Swap {
        _in_amount: amount_in,
        _minimum_out_amount: min_amount_out,
    }
    .serialize(&mut data)
    .unwrap();

    data
}

impl PoolFees {
    /// Calculate the trading fee in trading tokens
    pub fn trading_fee(&self, trading_tokens: u128) -> Option<u128> {
        calculate_fee(
            trading_tokens,
            u128::from(self.trade_fee_numerator),
            u128::from(self.trade_fee_denominator),
        )
    }

    /// Calculate the owner trading fee in trading tokens
    pub fn owner_trading_fee(&self, trading_tokens: u128) -> Option<u128> {
        calculate_fee(
            trading_tokens,
            u128::from(self.owner_trade_fee_numerator),
            u128::from(self.owner_trade_fee_denominator),
        )
    }
}

/// Helper function for calculating swap fee
pub fn calculate_fee(
    token_amount: u128,
    fee_numerator: u128,
    fee_denominator: u128,
) -> Option<u128> {
    if fee_numerator == 0 || token_amount == 0 {
        Some(0)
    } else {
        let fee = token_amount
            .checked_mul(fee_numerator)?
            .checked_div(fee_denominator)?;
        if fee == 0 {
            Some(1) // minimum fee of one token
        } else {
            Some(fee)
        }
    }
}
