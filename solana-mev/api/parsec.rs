pub struct ParsecApi {
    pub api_key: String,
    pub api_url: String,
}

impl ParsecApi {
    pub fn new(api_key: String, api_url: String) -> Self {
        ParsecApi {
            api_key,
            api_url,
        }
    }

    pub async fn get_token_prices(&self) -> Result<HashMap<String, f64>, reqwest::Error> {
        let url = format!("{}/prices", self.api_url);
        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            .header("Authorization", &self.api_key)
            .send()
            .await?;

        let prices: HashMap<String, f64> = response.json().await?;
        Ok(prices)
    }
}