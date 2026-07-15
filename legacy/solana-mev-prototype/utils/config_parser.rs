use crate::config::Config;
use anyhow::Result;
use std::fs::File;
use std::io::Read;

pub fn parse_config(path: &str) -> Result<Config> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let config: Config = toml::from_str(&contents)?;
    Ok(config)
}