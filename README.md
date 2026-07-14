# Solana Arbitrage Monitor

## Current status

The supported component is [`solana-mev`](solana-mev/README.md), a read-only
cyclic-arbitrage monitor using Jupiter Swap V2. It loads only a public taker
address, requests quotes, and never signs or submits transactions.

The `client-pool`, `arbitrage`, and `solana-program` directories are archived
experiments. They contain obsolete Serum, Orca Token Swap, direct DEX CPI, and
old Solana/Anchor dependencies. Do not deploy or fund them. See
[`MIGRATION.md`](MIGRATION.md) for the protocol and dependency audit.

## Run the supported monitor

Requirements:

- Rust 1.89 or newer
- A Jupiter API key from <https://portal.jup.ag>
- A public Solana wallet address

```bash
cd solana-mev
export JUPITER_API_KEY="..."
export SOLANA_TAKER_PUBKEY="your-public-wallet-address"
cargo run --release -- --once
```

Edit `solana-mev/config.toml` to configure routes. Token amounts and estimated
costs use each route's integer start-mint base units.

## Verification

```bash
cd solana-mev
cargo fmt --check
cargo check --all-targets --all-features --locked
cargo test --all-targets --all-features --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo audit
```

These checks validate the off-chain monitor. They do not certify the archived
on-chain CPI experiments or prove that any quoted opportunity can execute
profitably.

## Execution boundary

Before adding funded execution, both swap legs must be composed atomically,
simulated as the exact signed versioned transaction, and guarded by an on-chain
minimum-final-balance invariant. Profit calculations must include priority
fees, tips, Token-2022 transfer fees, account rent, and wrapping costs.
