use axum::{
    routing::get,
    Router,
    extract::State,
    http::StatusCode,
    Json,
};
use std::sync::Arc;
use std::net::SocketAddr;
use crate::price_fetcher::PriceFetcher;
use tower_http::cors::{CorsLayer, Any};

pub async fn start_api_server(price_fetcher: Arc<PriceFetcher>, port: u16) {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Create router with routes
    let app = Router::new()
        .route("/prices", get(get_all_prices))
        .route("/price/:token", get(get_token_price))
        .layer(cors)
        .with_state(price_fetcher);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("Starting API server on {}", addr);
    
    if let Err(e) = axum::serve(tokio::net::TcpListener::bind(addr).await.unwrap(), app).await {
        tracing::error!("Server error: {}", e);
    }
}

async fn get_all_prices(
    State(price_fetcher): State<Arc<PriceFetcher>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let prices = price_fetcher.get_all_prices();
    let json = serde_json::to_value(prices).unwrap_or(serde_json::Value::Null);
    (StatusCode::OK, Json(json))
}

async fn get_token_price(
    State(price_fetcher): State<Arc<PriceFetcher>>,
    axum::extract::Path(token_address): axum::extract::Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    match price_fetcher.get_price(&token_address) {
        Some(price) => {
            let json = serde_json::to_value(price).unwrap_or(serde_json::Value::Null);
            (StatusCode::OK, Json(json))
        },
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("Price for token {} not found", token_address)
            }))
        ),
    }
} 