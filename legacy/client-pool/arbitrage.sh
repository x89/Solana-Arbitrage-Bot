#!/bin/sh
set -eu

echo "Disabled: the legacy client must not submit mainnet transactions." >&2
echo "Run the observe-only monitor in ../solana-mev instead." >&2
exit 1