use crate::constants::*;
use crate::pool::PoolOperations;
use anchor_client::solana_sdk::pubkey::Pubkey;
use std::collections::HashMap;
use std::fs;
use std::rc::Rc;
use std::str::FromStr;

pub fn read_json_dir(dir: &String) -> Vec<String> {
    let _paths = fs::read_dir(dir).unwrap();
    let mut paths = Vec::new();
    for path in _paths {
        let p = path.unwrap().path();
        let path_str = p;
        match path_str.extension() {
            Some(ex) => {
                if ex == "json" {
                    let path = path_str.to_str().unwrap().to_string();
                    paths.push(path);
                }
            }
            None => {}
        }
    }
    paths
}

pub fn str2pubkey(s: &str) -> Pubkey {
    Pubkey::from_str(s).unwrap()
}

pub fn derive_token_address(owner: &Pubkey, mint: &Pubkey) -> Pubkey {
    let (pda, _) = Pubkey::find_program_address(
        &[
            &owner.to_bytes(),
            &TOKEN_PROGRAM_ID.to_bytes(),
            &mint.to_bytes(),
        ],
        &ASSOCIATED_TOKEN_PROGRAM_ID,
    );
    pda
}

#[derive(Debug, Clone)]
pub struct PoolQuote(pub Rc<Box<dyn PoolOperations>>);

impl PoolQuote {
    pub fn new(quote: Rc<Box<dyn PoolOperations>>) -> Self {
        Self(quote)
    }
}

#[derive(Debug)]
pub struct PoolGraph(pub HashMap<PoolIndex, PoolEdge>);

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct PoolIndex(pub usize);

#[derive(Debug, Clone)]
pub struct PoolEdge(pub HashMap<PoolIndex, Vec<PoolQuote>>);

impl PoolGraph {
    pub fn new() -> Self {
        Self(HashMap::new())
    }
}
