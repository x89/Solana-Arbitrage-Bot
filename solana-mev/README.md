# Solana Arbitrage Monitor

This directory is the canonical runnable component of the repository. It is a
dry-run cyclic-arbitrage monitor built against Jupiter Swap V2. Jupiter supplies
current routes and account layouts, so the monitor does not depend on stale
Raydium, Orca, or Meteora IDLs.

The old files outside `src/` are retained as legacy design material. They are
not compiled by this crate.

## Safety and scope

- No private key is loaded.
- No transaction is signed or submitted.
- A reported candidate is not a profit guarantee.
- The second quote is requested after the first, so the quotes are not atomic.
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

```bash
export JUPITER_API_KEY="..."
export SOLANA_TAKER_PUBKEY="your-public-wallet-address"

cargo run --release -- --once
# Or monitor continuously:
cargo run --release
```

Configure routes and risk assumptions in `config.toml`. All amounts are integer
base units; do not use floating-point UI amounts.

Useful checks:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Set `RUST_LOG=debug` for more logging.

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
