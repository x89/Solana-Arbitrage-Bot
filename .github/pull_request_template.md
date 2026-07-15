## Summary

<!-- Explain the user-visible outcome and why this change is needed. -->

## Verification

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo check --workspace --all-targets --all-features --locked`
- [ ] `cargo test --workspace --all-targets --all-features --locked`
- [ ] `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`
- [ ] Documentation and `CHANGELOG.md` updated when behavior changed
- [ ] No secrets, keypairs, funded execution, or legacy reactivation introduced

## Risk

<!-- Describe failure modes, rollback, configuration changes, and security impact. -->
