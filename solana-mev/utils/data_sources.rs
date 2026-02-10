use crate::error::Result;
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct PriceData {
    pub symbol: String,
    pub price: f64,
}

pub async fn fetch_prices_from_coingecko(symbols: &[String]) -> Result<HashMap<String, f64>> {
    let url = format!(
        "https://api.coingecko.com/api/v3/simple/price?ids={}&vs_currencies=usd",
        symbols.join(",")
    );
    
    let client = Client::new();
    let response = client.get(&url).send().await?;
    let prices: HashMap<String, HashMap<String, f64>> = response.json().await?;
    
    let mut price_map = HashMap::new();
    for (symbol, price_data) in prices {
        if let Some(price) = price_data.get("usd") {
            price_map.insert(symbol, *price);
        }
    }
    
    Ok(price_map)
}