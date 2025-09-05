use num_enum::{IntoPrimitive, TryFromPrimitive};
use std::convert::TryInto;
use anchor_client::solana_sdk::pubkey::Pubkey;

// FEE COMPUTATION 
mod stable_markets {
    pub mod usdt_usdc {
        solana_program::declare_id!("77quYg4MGneUdjgXCunt9GgM1usmrxKY31twEy3WHwcS");
    }
    pub mod msol_sol {
        solana_program::declare_id!("5cLrMai1DsLRYc1Nio9qMTicsWtvzjzZfJPXyAoF4t1Z");
    }
    pub mod ust_usdc {
        solana_program::declare_id!("EERNEEnBqdGzBS8dd46wwNY5F2kwnaCQ3vsq2fNKGogZ");
    }
    pub mod ust_usdt {
        solana_program::declare_id!("8sFf9TW3KzxLiBXcDcjAxqabEsRroo4EiRr3UG1xbJ9m");
    }
    pub mod stsol_sol {
        solana_program::declare_id!("2iDSTGhjJEiRxNaLF27CY6daMYPs5hgYrP2REHd5YD62");
    }
}

#[derive(Copy, Clone, IntoPrimitive, TryFromPrimitive, Debug)]
#[repr(u8)]
pub enum FeeTier {
    Base,
    SRM2,
    SRM3,
    SRM4,
    SRM5,
    SRM6,
    MSRM,
    Stable,
}

#[repr(transparent)]
#[derive(Copy, Clone)]
struct U64F64(u128);

impl U64F64 {
    const ONE: Self = U64F64(1 << 64);

    #[inline(always)]
    const fn add(self, other: U64F64) -> U64F64 {
        U64F64(self.0 + other.0)
    }

    #[inline(always)]
    const fn div(self, other: U64F64) -> u128 {
        self.0 / other.0
    }

    #[inline(always)]
    const fn mul_u64(self, other: u64) -> U64F64 {
        U64F64(self.0 * other as u128)
    }

    #[inline(always)]
    const fn floor(self) -> u64 {
        (self.0 >> 64) as u64
    }

    #[inline(always)]
    const fn frac_part(self) -> u64 {
        self.0 as u64
    }

    #[inline(always)]
    const fn from_int(n: u64) -> Self {
        U64F64((n as u128) << 64)
    }
}

#[inline(always)]
const fn fee_tenth_of_bps(tenth_of_bps: u64) -> U64F64 {
    U64F64(((tenth_of_bps as u128) << 64) / 100_000)
}

#[inline(always)]
const fn rebate_tenth_of_bps(tenth_of_bps: u64) -> U64F64 {
    U64F64(fee_tenth_of_bps(tenth_of_bps).0 + 1)
}

impl FeeTier {
    #[inline]
    pub fn from_srm_and_msrm_balances(market: &Pubkey, srm_held: u64, msrm_held: u64) -> FeeTier {
        let one_srm = 1_000_000;

        if market == &stable_markets::usdt_usdc::ID || market == &stable_markets::msol_sol::ID || market == &stable_markets::ust_usdc::ID || market == &stable_markets::ust_usdt::ID || market == &stable_markets::stsol_sol::ID {
            return FeeTier::Stable;
        }

        match () {
            () if msrm_held >= 1 => FeeTier::MSRM,
            () if srm_held >= one_srm * 1_000_000 => FeeTier::SRM6,
            () if srm_held >= one_srm * 100_000 => FeeTier::SRM5,
            () if srm_held >= one_srm * 10_000 => FeeTier::SRM4,
            () if srm_held >= one_srm * 1_000 => FeeTier::SRM3,
            () if srm_held >= one_srm * 100 => FeeTier::SRM2,
            () => FeeTier::Base,
        }
    }

    #[inline]
    pub fn maker_rebate(self, pc_qty: u64) -> u64 {
        rebate_tenth_of_bps(0).mul_u64(pc_qty).floor()
    }

    fn taker_rate(self) -> U64F64 {
        use FeeTier::*;
        match self {
            Base => fee_tenth_of_bps(40),
            SRM2 => fee_tenth_of_bps(39),
            SRM3 => fee_tenth_of_bps(38),
            SRM4 => fee_tenth_of_bps(36),
            SRM5 => fee_tenth_of_bps(34),
            SRM6 => fee_tenth_of_bps(32),
            MSRM => fee_tenth_of_bps(30),
            Stable => fee_tenth_of_bps(10),
        }
    }

    #[inline]
    pub fn taker_fee(self, pc_qty: u64) -> u64 {
        let rate = self.taker_rate();
        let exact_fee: U64F64 = rate.mul_u64(pc_qty);
        exact_fee.floor() + ((exact_fee.frac_part() != 0) as u64)
    }

    #[inline]
    pub fn remove_taker_fee(self, pc_qty_incl_fee: u64) -> u64 {
        let rate = self.taker_rate();
        U64F64::from_int(pc_qty_incl_fee)
            .div(U64F64::ONE.add(rate))
            .try_into()
            .unwrap()
    }
}

#[inline]
pub fn referrer_rebate(amount: u64) -> u64 {
    amount / 5
}