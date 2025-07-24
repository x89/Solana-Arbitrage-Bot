use anchor_gen::generate_cpi_crate;

generate_cpi_crate!("./idl.json");
declare_id!("24Uqj9JCLxUeoC3hGfh5W3s9FM9uCHDS2SG3LYwBpyTi");

pub const LOCKED_PROFIT_DEGRADATION_DENOMINATOR: u128 = 1_000_000_000_000;

impl LockedProfitTracker {
    pub fn calculate_locked_profit(&self, current_time: u64) -> Option<u64> {
        let duration = u128::from(current_time.checked_sub(self.last_report)?);
        let locked_profit_degradation = u128::from(self.locked_profit_degradation);
        let locked_fund_ratio = duration.checked_mul(locked_profit_degradation)?;

        if locked_fund_ratio > LOCKED_PROFIT_DEGRADATION_DENOMINATOR {
            return Some(0);
        }
        let locked_profit = u128::from(self.last_updated_locked_profit);

        let locked_profit = (locked_profit
            .checked_mul(LOCKED_PROFIT_DEGRADATION_DENOMINATOR - locked_fund_ratio)?)
        .checked_div(LOCKED_PROFIT_DEGRADATION_DENOMINATOR)?;
        let locked_profit = u64::try_from(locked_profit).ok()?;
        Some(locked_profit)
    }
}

impl Vault {
    pub fn get_amount_by_share(
        &self,
        current_time: u64,
        share: u64,
        total_supply: u64,
    ) -> Option<u64> {
        let total_amount = self.get_unlocked_amount(current_time)?;
        u64::try_from(
            u128::from(share)
                .checked_mul(u128::from(total_amount))?
                .checked_div(u128::from(total_supply))?,
        )
        .ok()
    }

    pub fn get_unlocked_amount(&self, current_time: u64) -> Option<u64> {
        self.total_amount.checked_sub(
            self.locked_profit_tracker
                .calculate_locked_profit(current_time)?,
        )
    }

    pub fn get_unmint_amount(
        &self,
        current_time: u64,
        out_token: u64,
        total_supply: u64,
    ) -> Option<u64> {
        let total_amount = self.get_unlocked_amount(current_time)?;
        u64::try_from(
            u128::from(out_token)
                .checked_mul(u128::from(total_supply))?
                .checked_div(u128::from(total_amount))?,
        )
        .ok()
    }
}
