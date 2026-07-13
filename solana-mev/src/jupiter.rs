use std::{collections::BTreeSet, time::Duration};

use anyhow::{bail, Context, Result};
use reqwest::{Client, StatusCode};
use serde::Deserialize;

#[derive(Clone)]
pub struct JupiterClient {
    http: Client,
    base_url: String,
    api_key: String,
}

#[derive(Debug, Clone)]
pub struct QuoteRequest<'a> {
    pub input_mint: &'a str,
    pub output_mint: &'a str,
    pub amount: u64,
    pub taker: &'a str,
    pub slippage_bps: u16,
    pub max_accounts: u8,
    pub fast_mode: bool,
    pub dexes: &'a [String],
}

#[derive(Debug, Clone)]
pub struct Quote {
    pub input_mint: String,
    pub output_mint: String,
    pub in_amount: u64,
    pub out_amount: u64,
    pub minimum_out_amount: u64,
    pub venue_labels: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BuildResponse {
    input_mint: String,
    output_mint: String,
    in_amount: String,
    out_amount: String,
    other_amount_threshold: String,
    #[serde(default)]
    route_plan: Vec<RoutePlanStep>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RoutePlanStep {
    swap_info: SwapInfo,
}

#[derive(Debug, Deserialize)]
struct SwapInfo {
    label: String,
}

impl JupiterClient {
    pub fn new(base_url: &str, api_key: String, timeout: Duration) -> Result<Self> {
        let http = Client::builder()
            .timeout(timeout)
            .user_agent(concat!(
                "solana-arbitrage-monitor/",
                env!("CARGO_PKG_VERSION")
            ))
            .build()
            .context("failed to construct HTTP client")?;

        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_owned(),
            api_key,
        })
    }

    pub async fn quote(&self, request: QuoteRequest<'_>) -> Result<Quote> {
        let endpoint = format!("{}/build", self.base_url);
        let mut query = vec![
            ("inputMint", request.input_mint.to_owned()),
            ("outputMint", request.output_mint.to_owned()),
            ("amount", request.amount.to_string()),
            ("taker", request.taker.to_owned()),
            ("slippageBps", request.slippage_bps.to_string()),
            ("maxAccounts", request.max_accounts.to_string()),
        ];

        if request.fast_mode {
            query.push(("mode", "fast".to_owned()));
        }
        if !request.dexes.is_empty() {
            query.push(("dexes", request.dexes.join(",")));
        }

        let response = self
            .http
            .get(endpoint)
            .header("x-api-key", &self.api_key)
            .query(&query)
            .send()
            .await
            .context("Jupiter request failed")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("failed to read Jupiter response")?;
        if !status.is_success() {
            return Err(api_error(status, &body));
        }

        let response: BuildResponse =
            serde_json::from_str(&body).context("Jupiter returned an unexpected response")?;
        let in_amount = parse_amount("inAmount", &response.in_amount)?;
        let out_amount = parse_amount("outAmount", &response.out_amount)?;
        let minimum_out_amount =
            parse_amount("otherAmountThreshold", &response.other_amount_threshold)?;

        if response.input_mint != request.input_mint
            || response.output_mint != request.output_mint
            || in_amount != request.amount
        {
            bail!("Jupiter response does not match the requested swap");
        }
        if out_amount == 0 || minimum_out_amount == 0 {
            bail!("Jupiter returned a zero output amount");
        }

        let venue_labels = response
            .route_plan
            .into_iter()
            .map(|step| step.swap_info.label)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        Ok(Quote {
            input_mint: response.input_mint,
            output_mint: response.output_mint,
            in_amount,
            out_amount,
            minimum_out_amount,
            venue_labels,
        })
    }
}

fn parse_amount(field: &str, value: &str) -> Result<u64> {
    value
        .parse()
        .with_context(|| format!("Jupiter {field} is not a valid u64"))
}

fn api_error(status: StatusCode, body: &str) -> anyhow::Error {
    let message = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .and_then(|error| error.as_str())
                .or_else(|| value.get("message").and_then(|message| message.as_str()))
                .map(str::to_owned)
        })
        .unwrap_or_else(|| body.chars().take(500).collect());
    anyhow::anyhow!("Jupiter API returned {status}: {message}")
}
