# Legacy pool client

This Anchor 0.22/Solana 1.9 prototype is retained for reference and is not the
supported bot. Its Serum and legacy pool decoders do not represent current
mainnet liquidity. Use `../solana-mev` for the maintained dry-run monitor.

The source no longer embeds an RPC credential or wallet path. If investigating
this legacy client, provide them through:

```bash
export SOLANA_RPC_URL="https://your-provider.example"
export SOLANA_KEYPAIR_PATH="/absolute/path/to/keypair.json"
```

Do not run it with a funded mainnet keypair.