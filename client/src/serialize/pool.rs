use std::fmt::Debug;
use serde;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct JSONFeeStructure {
    pub trader_fee: Fraction,
    pub owner_fee: Fraction,
}

#[derive(Deserialize, Serialize, Debug, Clone)]

pub struct Fraction {
    pub numerator: u64,
    pub denominator: u64,
}
