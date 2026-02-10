# Solana Raydium Sniper Bot

A Rust-based trading bot for Solana blockchain that monitors and executes trades based on specified target prices.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (1.70.0 or higher)
- [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) (comes with Rust)
- Solana wallet with funds

## Quick Start

1. **Clone and Install**

2. **Configure Environment**
   
   Create a `.env` file in the root directory:
   ```env
   TARGET_PRICE=0.000000000000001
   RPC_URL=https://mainnet.helius-rpc.com/?api-key=YOUR_API_KEY
   ```

3. **Build and Run**
   ```bash
   # Build the project
   cargo build

   # Run in development mode
   cargo run

   # Or build and run in release mode for better performance
   cargo build --release
   cargo run --release
   ```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TARGET_PRICE` | The target price for trading | Yes |
| `TARGET_ADDRESS` | Your target pool address | Yes |
| `RPC_URL` | Solana RPC endpoint URL | Yes |
