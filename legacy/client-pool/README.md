# Legacy pool client

This Anchor 0.22/Solana 1.9 prototype is retained for reference and is not the
supported bot. Its Serum and legacy pool decoders do not represent current
mainnet liquidity. Use [`../../solana-mev`](../../solana-mev) for the maintained
dry-run monitor.

The primary entry point no longer embeds an RPC credential. Dormant utilities
still contain obsolete fallback paths, and the crate does not currently build.
If investigating it, provide credentials and paths through:

```bash
export SOLANA_RPC_URL="https://your-provider.example"
export SOLANA_KEYPAIR_PATH="/absolute/path/to/keypair.json"
```

Do not run it with a funded mainnet keypair.