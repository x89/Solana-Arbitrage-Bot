# 2026 migration notes

## Decision

The supported path is off-chain discovery through Jupiter Swap V2. Direct DEX
CPIs remain isolated legacy experiments because their account layouts and
Anchor compatibility requirements differ by protocol.

This is safer than compiling old IDLs against a new Solana SDK: a successful
compile would not prove that account ordering, token-program selection, tick/bin
arrays, or fee math still match mainnet.

## Current toolchain and APIs

- Rust: 1.89 minimum; tested with 1.97
- Solana address crate: `solana-pubkey` 4.2
- Jupiter: `https://api.jup.ag/swap/v2/build`
- Anchor latest stable reviewed: 1.0.2
- Anchor's recommended Solana toolchain: 3.1.10

The runnable monitor intentionally does not require Anchor or the Solana CLI.

## Protocol findings

### Jupiter

Legacy Metis Swap V1 and Ultra are superseded by Swap V2. Swap V2 requires an
API key and uses `bps` as the canonical route split field. The monitor consumes
the current `/build` response instead of maintaining copied IDLs.

Jupiter Aggregator V6 program:
`JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4`.

### Raydium

Do not treat one `raydium_program_id` as covering all pools:

- AMM v4: `675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8`
- CPMM: `CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C`
- CLMM: `CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK`

CPMM is the recommended constant-product integration and supports Token-2022.
The old pool client incorrectly used the associated-token program as its
Raydium program constant.

### Orca

Legacy Orca Token Swap must not be used for new routes. The current Whirlpool
program is:
`whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc`.

### Meteora

The current DLMM program is:
`LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo`.

DLMM swaps require current bin-array accounts. The old generic reserve-based
pool abstraction cannot safely quote DLMM liquidity.

### Serum/OpenBook

Serum program `9xQe...` is obsolete. New order-book work must use a verified
OpenBook integration. No Serum account decoder is part of the supported path.

### Jito

Jito bundle submission remains useful for atomic landing, but it is not an RPC
URL with a source-controlled access token. The exposed token and absolute
wallet path were removed from the legacy pool client. The old token should be
revoked by its owner.

## Legacy workspace build status

- `solana-mev`: previously had no Cargo binary/library target; `cargo check`
  only checked dependency resolution. It now has a real `src/` target.
- `client-pool`: broken path dependency and obsolete Solana/Anchor 0.22 stack.
- `arbitrage`: dependency graph mixes Anchor 0.28/0.29 and Solana 1.16/2.0,
  producing incompatible `Pubkey` types.
- `solana-program`: host `cargo check` succeeds on its pinned legacy stack, but
  that does not validate mainnet CPI behavior.

## Before adding execution

1. Build both legs into one versioned transaction.
2. Use protocol-returned instructions and address lookup tables.
3. Simulate the exact signed message against the target RPC.
4. Compare pre/post token balances rather than trusting quoted values.
5. Include priority fee, Jito tip, Token-2022 transfer fees, ATA rent, and
   wrapping costs in the profit invariant.
6. Add an on-chain minimum-final-balance assertion.
7. Submit in dry-run/shadow mode before allowing funded execution.
