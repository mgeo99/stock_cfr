use chrono::NaiveDateTime;

/// Tick containing information relating to the current stock price
#[derive(Debug, PartialEq)]
pub struct StockQuote {
    pub timestamp: NaiveDateTime,
    pub open: f32,
    pub close: f32,
    pub high: f32,
    pub low: f32,
    pub volume: f32,
    pub symbol: String,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StockAction {
    /// Hold your current position
    Hold,
    /// Buy additional shares
    Buy(usize),
    /// Sell shares
    Sell(usize),
}
