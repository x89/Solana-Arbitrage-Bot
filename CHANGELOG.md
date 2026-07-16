# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and uses semantic versioning for the supported monitor.

## [Unreleased]

### Added

- Root Cargo workspace and pinned minimum Rust toolchain.
- CI gates for formatting, compilation, tests, Clippy, dependency policy,
  vulnerabilities, licenses, and secret scanning.
- Reproducible tagged release workflow with checksums and build provenance.
- Security policy, contribution guide, ownership rules, and dependency updates.
- Mock-backed HTTP and scanner integration coverage with no live-service
  dependency.
- Machine-readable JSON logs, scan identifiers, latency fields, and scan
  summaries.
- Observation-only Jupiter Swap V2 cyclic-arbitrage monitor.
- Strict configuration and quote-response validation.
- Conservative two-leg evaluation using the forward minimum output.
- Request throttling, bounded response reads, retry handling, and unit tests.

### Changed

- Moved obsolete clients, Anchor programs, prototypes, and conceptual diagrams
  under explicit legacy archives.
- Added an explicit configuration schema version and quote-cycle freshness
  deadline.
- Active scans now cancel cleanly on shutdown and stop immediately after denied
  API access.
- Rate-limit handling prefers Jupiter's `x-ratelimit-reset` timestamp and
  supports both delta-seconds and HTTP-date `Retry-After` fallbacks.
- Quote validation now rejects malformed AMM keys and venues outside an
  explicitly requested DEX set.

### Security

- Disabled legacy transaction and deployment paths.
- Removed hardcoded credential paths and expanded sensitive-file ignores.

[Unreleased]: https://github.com/x89/Solana-Arbitrage-Bot/commits/master
