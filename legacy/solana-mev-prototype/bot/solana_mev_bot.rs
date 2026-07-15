use crate::dex::dex_integration::DexIntegration;
use crate::models::market_conditions::MarketConditions;
use crate::models::mev_opportunity::MevOpportunity;
use crate::models::transaction_log::TransactionLog;
use crate::strategies::strategy::Strategy;
use solana_client::rpc_client::RpcClient;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::transaction::Transaction;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct SolanaMevBot {
    rpc_client: RpcClient,
    payer_keypair: Keypair,
    target_accounts: HashMap<Pubkey, AccountInfo>,
    profit_threshold: f64,
    dex_integrations: Vec<Box<dyn DexIntegration>>,
    flashbots_client: Arc<Mutex<FlashbotsClient>>,
    simulation_engine: Arc<Mutex<SimulationEngine>>,
    optimizer: Arc<Mutex<Optimizer>>,
    risk_manager: Arc<Mutex<RiskManager>>,
    market_analyzer: Arc<Mutex<MarketAnalyzer>>,
    strategy_manager: Arc<Mutex<StrategyManager>>,
    monitoring_manager: Arc<Mutex<MonitoringManager>>,
    copy_trade_manager: Arc<Mutex<CopyTradeManager>>,
    sniping_manager: Arc<Mutex<SnipingManager>>,
    gas_optimizer: Arc<Mutex<GasOptimizer>>,
    path_finder: Arc<Mutex<PathFinder>>,
    trade_executor: Arc<Mutex<TradeExecutor>>,
    cross_chain_manager: Arc<Mutex<CrossChainManager>>,
    order_manager: Arc<Mutex<OrderManager>>,
}

impl SolanaMevBot {
    pub async fn new(
        rpc_url: &str,
        payer_keypair: Keypair,
        target_accounts: HashMap<Pubkey, AccountInfo>,
        profit_threshold: f64,
        dex_integrations: Vec<Box<dyn DexIntegration>>,
    ) -> Self {
        let rpc_client = RpcClient::new(rpc_url.to_string());
        let flashbots_client = Arc::new(Mutex::new(FlashbotsClient::new(rpc_client.clone())));
        let simulation_engine = Arc::new(Mutex::new(SimulationEngine::new(rpc_client.clone())));
        let optimizer = Arc::new(Mutex::new(Optimizer::new(rpc_client.clone())));
        let risk_manager = Arc::new(Mutex::new(RiskManager::new(rpc_client.clone())));
        let market_analyzer = Arc::new(Mutex::new(MarketAnalyzer::new(rpc_client.clone())));
        let strategy_manager = Arc::new(Mutex::new(StrategyManager::new(rpc_client.clone(), dex_integrations.clone())));
        let monitoring_manager = Arc::new(Mutex::new(MonitoringManager::new(rpc_client.clone())));
        let copy_trade_manager = Arc::new(Mutex::new(CopyTradeManager::new(rpc_client.clone())));
        let sniping_manager = Arc::new(Mutex::new(SnipingManager::new(rpc_client.clone())));
        let gas_optimizer = Arc::new(Mutex::new(GasOptimizer::new(rpc_client.clone())));
        let path_finder = Arc::new(Mutex::new(PathFinder::new(rpc_client.clone())));
        let trade_executor = Arc::new(Mutex::new(TradeExecutor::new(rpc_client.clone())));
        let cross_chain_manager = Arc::new(Mutex::new(CrossChainManager::new(rpc_client.clone())));
        let order_manager = Arc::new(Mutex::new(OrderManager::new(rpc_client.clone())));

        SolanaMevBot {
            rpc_client,
            payer_keypair,
            target_accounts,
            profit_threshold,
            dex_integrations,
            flashbots_client,
            simulation_engine,
            optimizer,
            risk_manager,
            market_analyzer,
            strategy_manager,
            monitoring_manager,
            copy_trade_manager,
            sniping_manager,
            gas_optimizer,
            path_finder,
            trade_executor,
            cross_chain_manager,
            order_manager,
        }
    }

    pub async fn run(&mut self) {
        loop {
            let market_conditions = self.market_analyzer.lock().await.analyze().await;
            self.strategy_manager.lock().await.update(&market_conditions);
            self.risk_manager.lock().await.update(&market_conditions);
            self.copy_trade_manager.lock().await.update(&market_conditions);
            self.sniping_manager.lock().await.update(&market_conditions);
            self.gas_optimizer.lock().await.update(&market_conditions);
            self.path_finder.lock().await.update(&market_conditions);
            self.cross_chain_manager.lock().await.update(&market_conditions);
            self.order_manager.lock().await.update(&market_conditions);

            let opportunities = self.find_opportunities().await;
            let mut all_opportunities = Vec::new();
            all_opportunities.extend(opportunities);

            let copy_trade_opportunities = self.copy_trade_manager.lock().await.find_opportunities().await;
            all_opportunities.extend(copy_trade_opportunities);

            let sniping_opportunities = self.sniping_manager.lock().await.find_opportunities().await;
            all_opportunities.extend(sniping_opportunities);

            let cross_chain_opportunities = self.cross_chain_manager.lock().await.find_opportunities().await;
            all_opportunities.extend(cross_chain_opportunities);

            let profitable_txns = self.optimize_and_filter_txns(&all_opportunities).await;
            let gas_optimized_txns = self.gas_optimizer.lock().await.optimize(&profitable_txns).await;

            let executed_txns = self.trade_executor.lock().await.execute_transactions(&gas_optimized_txns).await;
            self.monitoring_manager.lock().await.log_and_monitor(&executed_txns, &market_conditions);

            self.order_manager.lock().await.manage_orders(&executed_txns).await;

            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    }

    async fn find_opportunities(&self) -> Vec<MevOpportunity> {
        let mut opportunities = Vec::new();

        for dex_integration in &self.dex_integrations {
            let dex_opportunities = dex_integration.find_opportunities(
                &self.target_accounts,
                &self.market_analyzer,
                &self.strategy_manager,
            ).await;
            opportunities.extend(dex_opportunities);
        }

        let path_opportunities = self.path_finder.lock().await.find_opportunities(&self.target_accounts).await;
        opportunities.extend(path_opportunities);

        opportunities
    }

    async fn optimize_and_filter_txns(&self, opportunities: &[MevOpportunity]) -> Vec<Transaction> {
        let mut profitable_txns = Vec::new();

        for opportunity in opportunities {
            let optimized_txns = self.optimizer.lock().await.optimize(opportunity).await;
            for tx in &optimized_txns {
                if self.risk_manager.lock().await.is_safe(tx).await && self.is_profitable(tx).await {
                    profitable_txns.push(tx.clone());
                }
            }
        }

        profitable_txns
    }

    async fn is_profitable(&self, tx: &Transaction) -> bool {
        let profit = self.simulation_engine.lock().await.simulate(tx).await;
        profit >= self.profit_threshold
    }
}