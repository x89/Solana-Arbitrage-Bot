# Contributing

Contributions are welcome to the supported observation-only monitor in
`solana-mev`. Archived code under `legacy/` is not a product surface and should
not be revived through dependency-only changes.

## Setup

Install `rustup`; the repository's `rust-toolchain.toml` installs Rust 1.89,
Clippy, and rustfmt. Then run:

```bash
cargo check --workspace --all-targets --all-features --locked
cargo test --workspace --all-targets --all-features --locked
```

Install the two repository-policy tools before running the complete local gate:

```bash
cargo install cargo-audit --locked
cargo install cargo-deny --locked
```

Copy only the environment variables you need from `.env.example`. The
application does not read `.env` files automatically.

## Required checks

Before opening a pull request:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features --locked
cargo test --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo audit
cargo deny check
```

CI runs formatting, compilation, tests, and strict Clippy on the minimum
supported Rust version; latest stable runs compilation and tests. Separate jobs
run RustSec, cargo-deny, and secret scanning. Add unit or integration coverage
for every behavior change. Tests must not depend on the public Jupiter service,
funded wallets, or mainnet state.

## Pull requests

Keep changes focused. Explain the outcome, operational risk, configuration
impact, and verification performed. Update `CHANGELOG.md` and documentation for
user-visible changes.

Never commit credentials, keypair JSON, private RPC URLs, generated build
artifacts, or real wallet identifiers. Report security issues according to
`SECURITY.md`.

## Scope boundary

Transaction construction, signing, bundle submission, and on-chain programs
are outside this monitor's scope. Proposals for funded execution require a
separate architecture and security review; they must not reuse archived code
as though it were protocol-correct.
