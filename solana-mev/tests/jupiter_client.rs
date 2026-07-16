use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mev_bot_solana::jupiter::{api_error_details, JupiterClient, QuoteRequest};
use serde_json::json;
use wiremock::{
    matchers::{header, method, path, query_param},
    Mock, MockServer, ResponseTemplate,
};

const INPUT_MINT: &str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";
const OUTPUT_MINT: &str = "So11111111111111111111111111111111111111112";
const TAKER: &str = "11111111111111111111111111111111";
const AMM_KEY: &str = "11111111111111111111111111111111";

fn request() -> QuoteRequest<'static> {
    QuoteRequest {
        input_mint: INPUT_MINT,
        output_mint: OUTPUT_MINT,
        amount: 100_000_000,
        taker: TAKER,
        slippage_bps: 30,
        max_accounts: 64,
        fast_mode: true,
        dexes: &[],
    }
}

fn quote_body() -> serde_json::Value {
    json!({
        "inputMint": INPUT_MINT,
        "outputMint": OUTPUT_MINT,
        "inAmount": "100000000",
        "outAmount": "750000000",
        "otherAmountThreshold": "747750000",
        "swapMode": "ExactIn",
        "slippageBps": 30,
        "routePlan": [{
            "swapInfo": {
                "ammKey": AMM_KEY,
                "label": "Raydium CLMM"
            }
        }]
    })
}

fn client(server: &MockServer, minimum_interval: Duration) -> JupiterClient {
    JupiterClient::new(
        &server.uri(),
        "test-api-key".to_owned(),
        Duration::from_secs(2),
        minimum_interval,
    )
    .unwrap()
}

#[tokio::test]
async fn sends_expected_request_and_parses_quote() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/build"))
        .and(header("x-api-key", "test-api-key"))
        .and(query_param("inputMint", INPUT_MINT))
        .and(query_param("outputMint", OUTPUT_MINT))
        .and(query_param("amount", "100000000"))
        .and(query_param("taker", TAKER))
        .and(query_param("slippageBps", "30"))
        .and(query_param("maxAccounts", "64"))
        .and(query_param("mode", "fast"))
        .respond_with(ResponseTemplate::new(200).set_body_json(quote_body()))
        .expect(1)
        .mount(&server)
        .await;

    let quote = client(&server, Duration::ZERO)
        .quote(request())
        .await
        .unwrap();

    assert_eq!(quote.minimum_out_amount, 747_750_000);
    assert_eq!(quote.venue_labels, ["Raydium CLMM"]);
    assert_eq!(quote.amm_keys, [AMM_KEY]);
}

#[tokio::test]
async fn exposes_rate_limit_retry_metadata() {
    let server = MockServer::start().await;
    let reset_timestamp = (SystemTime::now() + Duration::from_secs(60))
        .duration_since(UNIX_EPOCH)
        .expect("current time after Unix epoch")
        .as_secs();
    Mock::given(method("GET"))
        .and(path("/build"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("x-ratelimit-reset", reset_timestamp.to_string())
                .set_body_json(json!({"error": "rate limited"})),
        )
        .mount(&server)
        .await;

    let error = client(&server, Duration::ZERO)
        .quote(request())
        .await
        .unwrap_err();
    let details = api_error_details(&error).expect("typed API error");

    assert!(!details.is_permanent());
    let retry_after = details.retry_after().expect("rate-limit reset metadata");
    assert!(retry_after >= Duration::from_secs(55));
    assert!(retry_after <= Duration::from_secs(60));
}

#[tokio::test]
async fn parses_http_date_retry_after() {
    let server = MockServer::start().await;
    let retry_at = SystemTime::now() + Duration::from_secs(60);
    Mock::given(method("GET"))
        .and(path("/build"))
        .respond_with(
            ResponseTemplate::new(503)
                .insert_header("Retry-After", httpdate::fmt_http_date(retry_at))
                .set_body_json(json!({"error": "temporarily unavailable"})),
        )
        .mount(&server)
        .await;

    let error = client(&server, Duration::ZERO)
        .quote(request())
        .await
        .unwrap_err();
    let retry_after = api_error_details(&error)
        .and_then(|details| details.retry_after())
        .expect("HTTP-date retry metadata");

    assert!(retry_after >= Duration::from_secs(55));
    assert!(retry_after <= Duration::from_secs(60));
}

#[tokio::test]
async fn refuses_redirects_and_oversized_responses() {
    let redirect_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/build"))
        .respond_with(
            ResponseTemplate::new(302)
                .insert_header("Location", format!("{}/target", redirect_server.uri())),
        )
        .mount(&redirect_server)
        .await;
    Mock::given(method("GET"))
        .and(path("/target"))
        .respond_with(ResponseTemplate::new(200).set_body_json(quote_body()))
        .expect(0)
        .mount(&redirect_server)
        .await;

    let error = client(&redirect_server, Duration::ZERO)
        .quote(request())
        .await
        .unwrap_err();
    assert!(api_error_details(&error).is_some());

    let large_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/build"))
        .respond_with(ResponseTemplate::new(200).set_body_string("x".repeat(2 * 1024 * 1024 + 1)))
        .mount(&large_server)
        .await;

    let error = client(&large_server, Duration::ZERO)
        .quote(request())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("exceeds"));
}

#[tokio::test]
async fn serializes_requests_at_configured_interval() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/build"))
        .respond_with(ResponseTemplate::new(200).set_body_json(quote_body()))
        .expect(2)
        .mount(&server)
        .await;
    let client = client(&server, Duration::from_millis(40));

    let started = Instant::now();
    client.quote(request()).await.unwrap();
    client.quote(request()).await.unwrap();

    assert!(started.elapsed() >= Duration::from_millis(35));
}
