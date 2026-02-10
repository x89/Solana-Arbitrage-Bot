use crate::api::parsec::ParsecApi;
use crate::api::flipside::FlipsideApi;
use crate::api::thegraph::TheGraphApi;

pub async fn get_sniping_data(parsec_api: &ParsecApi, flipside_api: &FlipsideApi) -> Result<crate::models::sniping_data::SnipingData, Box<dyn std::error::Error>> {
    let token_prices = parsec_api.get_token_prices().await?;
    
    let mut sniping_data = crate::models::sniping_data::SnipingData {
        token_prices,
        token_volumes: HashMap::new(),
    };
    
    for token_mint in token_prices.keys() {
        let token_volume = flipside_api.get_token_volume(token_mint).await?;
        sniping_data.token_volumes.insert(token_mint.to_string(), token_volume);
    }
    
    Ok(sniping_data)
}

pub async fn get_copy_trade_data(thegraph_api: &TheGraphApi, trader_accounts: &[String]) -> Result<Vec<crate::models::trade::Trade>, Box<dyn std::error::Error>> {
    let mut all_trades = Vec::new();
    
    for trader_account in trader_accounts {
        let trades = thegraph_api.get_trader_transactions(trader_account).await?;
        all_trades.extend(trades);
    }
    
    Ok(all_trades)
}