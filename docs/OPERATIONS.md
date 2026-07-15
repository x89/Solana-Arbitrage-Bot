# Operations runbook

## Repository settings

The repository owner must configure controls that source files cannot enable:

1. Protect the default branch and require pull requests.
2. Require the `Rust 1.89 quality gates`, `Latest stable compatibility`,
   `Dependency and license policy`, and `Secret scan` checks.
3. Require CODEOWNER review and resolution of review conversations.
4. Block force pushes and branch deletion.
5. Enable private vulnerability reporting, Dependabot alerts, push protection,
   and GitHub secret scanning where the repository plan supports them.
6. Confirm at the provider that every historically exposed credential was
   revoked; retain the evidence privately.

Do not mark the credential action complete merely because the value is absent
from the current tree. Git history cleanup is not revocation.

## Pre-deployment

The monitor is observation-only and needs no funded wallet or Solana RPC.

1. Build from a reviewed release tag or verified provenance artifact.
2. Verify the `.sha256` file before installation.
3. Store `JUPITER_API_KEY` in the runtime secret manager.
4. Set `SOLANA_TAKER_PUBKEY` to an unfunded public address.
5. Validate configuration without credentials:

   ```bash
   solana-arbitrage-monitor \
     --config /etc/solana-arbitrage-monitor/config.toml \
     --validate-config
   ```

6. Start once with `--once --log-format json`; inspect errors and latency.
7. Run continuously only after confirming the configured Jupiter quota.

## Health signals

Each scan emits a summary with `scan_id`, route count, error count, candidate
count, and elapsed time. Each successful evaluation reports
`cycle_duration_ms`. Alert on:

- repeated scan errors or sustained backoff;
- authentication failures, which stop the process;
- cycle time approaching `max_cycle_duration_ms`;
- missing scan summaries;
- sustained rate limiting;
- unexpected candidate volume.

Candidate logs are research signals, not orders or profit guarantees.

## Shutdown and rollback

`SIGINT`/`Ctrl-C` cancels an active scan or inter-scan sleep. Stop the service
before changing secrets or configuration. Roll back by restoring the last
reviewed configuration and verified release artifact; validate it before
restart.

## Release procedure

1. Update `CHANGELOG.md` and the crate version in one reviewed pull request.
2. Ensure all required checks pass on the protected default branch.
3. Create a signed `v<crate-version>` Git tag.
4. Push the tag. The release workflow rejects a tag that does not match the
   crate version.
5. Verify both platform archives, checksums, and GitHub build-provenance
   attestations on the generated release.

The workflow publishes Linux x86-64 and Apple Silicon macOS binaries. Other
platforms must build from the same reviewed tag with the pinned toolchain.
