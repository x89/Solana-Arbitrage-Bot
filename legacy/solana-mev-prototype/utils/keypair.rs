use anyhow::Result;
use solana_sdk::signature::Keypair;
use std::fs::File;
use std::io::BufReader;

pub fn read_keypair_file(path: &str) -> Result<Keypair> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let keypair = Keypair::from_bytes(&serde_json::from_reader(reader)?)?;
    Ok(keypair)
}