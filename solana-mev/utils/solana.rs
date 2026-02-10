use solana_sdk::transaction::Transaction;
use std::error::Error;

pub fn analyze_transaction(transaction: &Transaction) -> Result<crate::models::transaction_analysis::TransactionAnalysis, Box<dyn Error>> {
    let mut analysis = crate::models::transaction_analysis::TransactionAnalysis::default();
    
    analysis.signature = transaction.signatures[0].to_string();
    analysis.num_instructions = transaction.message.instructions.len() as u64;
    
    for (index, instruction) in transaction.message.instructions.iter().enumerate() {
        let account_metas = &instruction.accounts;
        let num_accounts = account_metas.len() as u64;
        let program_id = &instruction.program_id;
        
        analysis.instructions.push(crate::models::transaction_analysis::InstructionAnalysis {
            index: index as u64,
            num_accounts,
            program_id: program_id.to_string(),
        });
    }
    
    Ok(analysis)
}

pub fn calculate_profit(transaction: &Transaction) -> Result<f64, Box<dyn Error>> {
    let mut profit = 0.0;
    
    for (index, instruction) in transaction.message.instructions.iter().enumerate() {
        let account_metas = &instruction.accounts;
        
        if let Some(transfer_instruction) = instruction.program_id(&spl_token::ID) {
            if let Ok(transfer_amount) = spl_token::instruction::unpack_amount(transfer_instruction.data) {
                let from_account = &account_metas[0];
                let to_account = &account_metas[1];
                
                if from_account.is_signer {
                    profit -= transfer_amount as f64;
                } else if to_account.is_signer {
                    profit += transfer_amount as f64;
                }
            }
        }
    }
    
    Ok(profit)
}