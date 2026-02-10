use crate::error::Result;
use solana_sdk::instruction::Instruction;
use solana_sdk::transaction::Transaction;

pub fn get_instruction_data(transaction: &Transaction, program_id: &Pubkey) -> Result<Vec<u8>> {
    let instruction = transaction
        .message
        .instructions
        .iter()
        .find(|ix| ix.program_id == *program_id)
        .ok_or_else(|| anyhow!("Instruction not found for program ID: {}", program_id))?;
    
    Ok(instruction.data.clone())
}

pub fn get_instruction_accounts(transaction: &Transaction, program_id: &Pubkey) -> Result<Vec<Pubkey>> {
    let instruction = transaction
        .message
        .instructions
        .iter()
        .find(|ix| ix.program_id == *program_id)
        .ok_or_else(|| anyhow!("Instruction not found for program ID: {}", program_id))?;
    
    Ok(instruction.accounts.iter().map(|account| account.pubkey).collect())
}