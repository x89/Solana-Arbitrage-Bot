# Solana Arbitrage Monitor

This directory is the canonical runnable component of the repository. It is a
dry-run cyclic-arbitrage monitor built against Jupiter Swap V2. Jupiter supplies
current routes and account layouts, so the monitor does not depend on stale
Raydium, Orca, or Meteora IDLs.

Historical files were moved to `../legacy/solana-mev-prototype`; only `src/`
and the integration tests in this directory belong to the maintained crate.

## Safety and scope

- No private key is loaded.
- No transaction is signed or submitted.
- A reported candidate is not a profit guarantee.
- The second quote is requested after the first, so the quotes are not atomic.
- The return quote uses the forward leg's minimum output, and requests are
  serialized according to `jupiter.min_request_interval_ms`.
- The complete two-quote cycle is cancelled when it reaches
  `scanner.max_cycle_duration_ms`.
- Execution must simulate a single composed transaction and include priority
  fees, Jito tips, token transfer fees, and account-rent costs.

This monitor deliberately does not implement sandwich trading or front-running.

## Requirements

- Rust 1.89 or newer
- A Jupiter API key from <https://portal.jup.ag>
- A public Solana wallet address used as Jupiter's `taker`

The crate uses Jupiter's current `https://api.jup.ag/swap/v2/build` endpoint and
the current modular Solana address crate instead of the retired monolithic
Solana 1.x SDK.

## Run

From the repository root:

```bash
export JUPITER_API_KEY="..."
export SOLANA_TAKER_PUBKEY="your-public-wallet-address"

cargo run --release --package mev-bot-solana -- \
  --config solana-mev/config.toml --once
# Or monitor continuously:
cargo run --release --package mev-bot-solana -- \
  --config solana-mev/config.toml
```

Configure routes and risk assumptions in `config.toml`. All amounts are integer
base units; do not use floating-point UI amounts. The per-route cost allowance
is an operator estimate, not proof that all execution costs are covered.
Configuration requires `schema_version = 1`.

Validate configuration without credentials or a network request:

```bash
cargo run --package mev-bot-solana -- \
  --config solana-mev/config.toml --validate-config
```

Use `--log-format json` for structured log ingestion and `RUST_LOG=debug` for
request latency details.

Useful checks:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features --locked
cargo test --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
```

## Current protocol boundary

Jupiter Swap V2 performs route discovery across supported Solana liquidity
venues and returns the actual program instructions and address lookup tables.
This avoids copying pool layouts and program IDs into the scanner.

Direct on-chain CPI execution is intentionally separate. Each protocol must be
migrated and tested independently:

- Raydium CPMM/CLMM uses different programs and layouts from legacy AMM v4.
- Orca integrations should target Whirlpools, not the retired Orca Token Swap.
- Meteora DLMM requires its current bin-array accounts and CPI interface.
- Serum is obsolete and must not be used for new integrations.
