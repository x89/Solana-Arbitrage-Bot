pub struct TheGraphApi {
    pub api_url: String,
}

impl TheGraphApi {
    pub fn new(api_url: String) -> Self {
        TheGraphApi {
            api_url,
        }
    }

    pub async fn get_trader_transactions(&self, trader_account: &str) -> Result<Vec<crate::models::trade::Trade>, reqwest::Error> {
        let query = format!("{{ traderTransactions(trader: \"{}\") {{ id tokenAmount tokenMint }} }}", trader_account);
        let client = reqwest::Client::new();
        let response = client
            .post(&self.api_url)
            .json(&serde_json::json!({ "query": query }))
            .send()
            .await?;

        let result: serde_json::Value = response.json().await?;
        let transactions = result["data"]["traderTransactions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|trade| crate::models::trade::Trade {
                id: trade["id"].as_str().unwrap().to_string(),
                token_amount: trade["tokenAmount"].as_f64().unwrap(),
                token_mint: trade["tokenMint"].as_str().unwrap().to_string(),
            })
            .collect();

        Ok(transactions)
    }
}