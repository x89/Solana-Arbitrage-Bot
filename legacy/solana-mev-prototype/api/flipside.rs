pub struct FlipsideApi {
    pub api_key: String,
    pub api_url: String,
}

impl FlipsideApi {
    pub fn new(api_key: String, api_url: String) -> Self {
        FlipsideApi {
            api_key,
            api_url,
        }
    }

    pub async fn get_token_volume(&self, token_mint: &str) -> Result<f64, reqwest::Error> {
        let url = format!("{}/volume?token_mint={}", self.api_url, token_mint);
        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            .header("Authorization", &self.api_key)
            .send()
            .await?;

        let volume: f64 = response.json().await?;
        Ok(volume)
    }
}