use crate::error::Result;
use crate::models::market::Market;
use crate::models::order::{Order, OrderSide};
use rust_decimal::Decimal;

pub fn calculate_profit(market: &Market, buy_order: &Order, sell_order: &Order) -> Result<f64> {
    if buy_order.side != OrderSide::Bid || sell_order.side != OrderSide::Ask {
        return Err(anyhow!("Invalid order sides for profit calculation"));
    }
    
    let buy_price = Decimal::from_f64(buy_order.price).ok_or_else(|| anyhow!("Invalid buy price"))?;
    let sell_price = Decimal::from_f64(sell_order.price).ok_or_else(|| anyhow!("Invalid sell price"))?;
    let quantity = Decimal::from_f64(buy_order.quantity).ok_or_else(|| anyhow!("Invalid quantity"))?;
    
    let buy_value = buy_price * quantity;
    let sell_value = sell_price * quantity;
    let profit = sell_value - buy_value;
    
    Ok(profit.to_f64().ok_or_else(|| anyhow!("Failed to convert profit to f64"))?)
}