use std::{collections::BTreeSet, fmt, sync::Arc, time::Duration};

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use reqwest::{header::RETRY_AFTER, Client, Response, StatusCode};
use serde::Deserialize;
use tokio::{sync::Mutex, time::Instant};
use tracing::debug;

const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone)]
pub struct JupiterClient {
    http: Client,
    base_url: String,
    api_key: String,
    min_request_interval: Duration,
    request_gate: Arc<Mutex<Option<Instant>>>,
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
    pub amm_keys: Vec<String>,
}

#[derive(Debug)]
pub struct JupiterApiError {
    status: StatusCode,
    retry_after: Option<Duration>,
    message: String,
}

impl JupiterApiError {
    pub fn is_permanent(&self) -> bool {
        matches!(
            self.status,
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN
        )
    }

    pub fn retry_after(&self) -> Option<Duration> {
        self.retry_after
    }
}

impl fmt::Display for JupiterApiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Jupiter API returned {}: {}",
            self.status, self.message
        )
    }
}

impl std::error::Error for JupiterApiError {}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BuildResponse {
    input_mint: String,
    output_mint: String,
    in_amount: String,
    out_amount: String,
    other_amount_threshold: String,
    swap_mode: String,
    slippage_bps: u16,
    #[serde(default)]
    route_plan: Vec<RoutePlanStep>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RoutePlanStep {
    swap_info: SwapInfo,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SwapInfo {
    amm_key: String,
    label: String,
}

impl JupiterClient {
    pub fn new(
        base_url: &str,
        api_key: String,
        timeout: Duration,
        min_request_interval: Duration,
    ) -> Result<Self> {
        let http = Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
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
            min_request_interval,
            request_gate: Arc::new(Mutex::new(None)),
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

        self.wait_for_request_slot().await;
        debug!(
            input_mint = request.input_mint,
            output_mint = request.output_mint,
            amount = request.amount,
            "requesting Jupiter quote"
        );
        let response = self
            .http
            .get(endpoint)
            .header("x-api-key", &self.api_key)
            .query(&query)
            .send()
            .await
            .context("Jupiter request failed")?;

        let status = response.status();
        let retry_after = parse_retry_after(&response);
        let body = read_bounded_body(response).await?;
        if !status.is_success() {
            return Err(api_error(status, retry_after, &body));
        }

        parse_quote(&body, &request)
    }

    async fn wait_for_request_slot(&self) {
        let mut last_request = self.request_gate.lock().await;
        if let Some(last_request) = *last_request {
            let elapsed = last_request.elapsed();
            if elapsed < self.min_request_interval {
                tokio::time::sleep(self.min_request_interval - elapsed).await;
            }
        }
        *last_request = Some(Instant::now());
    }
}

fn parse_quote(body: &str, request: &QuoteRequest<'_>) -> Result<Quote> {
    let response: BuildResponse =
        serde_json::from_str(body).context("Jupiter returned an unexpected response")?;
    let in_amount = parse_amount("inAmount", &response.in_amount)?;
    let out_amount = parse_amount("outAmount", &response.out_amount)?;
    let minimum_out_amount =
        parse_amount("otherAmountThreshold", &response.other_amount_threshold)?;

    if response.input_mint != request.input_mint
        || response.output_mint != request.output_mint
        || in_amount != request.amount
        || response.swap_mode != "ExactIn"
        || response.slippage_bps != request.slippage_bps
    {
        bail!("Jupiter response does not match the requested swap");
    }
    if out_amount == 0 || minimum_out_amount == 0 {
        bail!("Jupiter returned a zero output amount");
    }
    if minimum_out_amount > out_amount {
        bail!("Jupiter minimum output exceeds its quoted output");
    }
    if response.route_plan.is_empty() {
        bail!("Jupiter returned a quote without a route plan");
    }

    let mut venue_labels = BTreeSet::new();
    let mut amm_keys = BTreeSet::new();
    for step in response.route_plan {
        let label = step.swap_info.label.trim();
        let amm_key = step.swap_info.amm_key.trim();
        if label.is_empty() || amm_key.is_empty() {
            bail!("Jupiter returned an incomplete route step");
        }
        venue_labels.insert(label.to_owned());
        amm_keys.insert(amm_key.to_owned());
    }

    Ok(Quote {
        input_mint: response.input_mint,
        output_mint: response.output_mint,
        in_amount,
        out_amount,
        minimum_out_amount,
        venue_labels: venue_labels.into_iter().collect(),
        amm_keys: amm_keys.into_iter().collect(),
    })
}

fn parse_amount(field: &str, value: &str) -> Result<u64> {
    value
        .parse()
        .with_context(|| format!("Jupiter {field} is not a valid u64"))
}

async fn read_bounded_body(response: Response) -> Result<String> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
    {
        bail!("Jupiter response exceeds {MAX_RESPONSE_BYTES} bytes");
    }

    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("failed to read Jupiter response")?;
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            bail!("Jupiter response exceeds {MAX_RESPONSE_BYTES} bytes");
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body).context("Jupiter response is not valid UTF-8")
}

fn parse_retry_after(response: &Response) -> Option<Duration> {
    response
        .headers()
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

pub fn api_error_details(error: &anyhow::Error) -> Option<&JupiterApiError> {
    error.downcast_ref()
}

fn api_error(status: StatusCode, retry_after: Option<Duration>, body: &str) -> anyhow::Error {
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
    JupiterApiError {
        status,
        retry_after,
        message,
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::{parse_quote, QuoteRequest};

    const INPUT_MINT: &str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";
    const OUTPUT_MINT: &str = "So11111111111111111111111111111111111111112";

    fn request() -> QuoteRequest<'static> {
        QuoteRequest {
            input_mint: INPUT_MINT,
            output_mint: OUTPUT_MINT,
            amount: 100_000_000,
            taker: "11111111111111111111111111111111",
            slippage_bps: 30,
            max_accounts: 64,
            fast_mode: true,
            dexes: &[],
        }
    }

    fn response(minimum_out: &str, route_plan: &str) -> String {
        format!(
            r#"{{
                "inputMint":"{INPUT_MINT}",
                "outputMint":"{OUTPUT_MINT}",
                "inAmount":"100000000",
                "outAmount":"750000000",
                "otherAmountThreshold":"{minimum_out}",
                "swapMode":"ExactIn",
                "slippageBps":30,
                "routePlan":{route_plan}
            }}"#
        )
    }

    #[test]
    fn parses_and_deduplicates_valid_route_labels() {
        let body = response(
            "747750000",
            r#"[
                {"swapInfo":{"ammKey":"pool-a","label":"Raydium CLMM"}},
                {"swapInfo":{"ammKey":"pool-a","label":"Raydium CLMM"}}
            ]"#,
        );
        let quote = parse_quote(&body, &request()).unwrap();

        assert_eq!(quote.minimum_out_amount, 747_750_000);
        assert_eq!(quote.venue_labels, ["Raydium CLMM"]);
        assert_eq!(quote.amm_keys, ["pool-a"]);
    }

    #[test]
    fn rejects_invalid_threshold_and_empty_route() {
        assert!(parse_quote(&response("750000001", "[]"), &request()).is_err());
        assert!(parse_quote(&response("747750000", "[]"), &request()).is_err());
        assert!(parse_quote(
            &response(
                "747750000",
                r#"[{"swapInfo":{"ammKey":"","label":"Raydium CLMM"}}]"#
            ),
            &request()
        )
        .is_err());
    }
}
