# Legacy multi-DEX Anchor experiment

Do not deploy or upgrade this program. Its placeholder program ID, nested CPI
account model, and mixed Anchor/Solana dependency graph are not valid for a
current mainnet deployment.

Anchor 1.0.2 targets Solana 3.1.10, but updating only these version strings is
unsafe: Raydium, Orca, and Meteora each require a protocol-specific CPI
migration and current account validation. See
[`../../MIGRATION.md`](../../MIGRATION.md).

The supported runnable component is
[`../../solana-mev`](../../solana-mev).