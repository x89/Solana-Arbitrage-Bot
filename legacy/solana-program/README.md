# Legacy swap-wrapper Anchor experiment

This pinned Anchor 0.28/Solana 1.16 workspace still passes a host `cargo check`,
but that does not establish that its CPI account lists or instruction data are
valid on current mainnet.

Do not deploy or upgrade it without independent migrations for Raydium CPMM or
CLMM, Orca Whirlpools, Meteora DLMM, and Jupiter Swap V2. Anchor 1.0.2 targets
Solana 3.1.10; a blanket dependency bump would conflict with the old protocol
CPI crates. See [`../../MIGRATION.md`](../../MIGRATION.md).

The supported runnable component is
[`../../solana-mev`](../../solana-mev).