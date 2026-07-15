# Security policy

## Supported surface

Only the latest release of `solana-mev` is supported. Everything under
`legacy/` is archived, intentionally excluded from builds, and unsafe for
deployment or funded use.

The supported monitor is observation-only. It must not load private keys,
sign transactions, submit bundles, or promise executable profit.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting:

<https://github.com/x89/Solana-Arbitrage-Bot/security/advisories/new>

Do not open a public issue containing a key, seed phrase, API token, private RPC
URL, wallet path, exploit proof, or other sensitive material. Include the
affected commit or release, impact, reproduction steps, and a suggested fix if
known. Allow maintainers a reasonable remediation window before disclosure.

## Credential incident response

Deleting a credential from Git history does not revoke it. If any credential
has ever appeared in a commit, log, screenshot, test fixture, or release:

1. Revoke it at the provider immediately.
2. Create a replacement with the minimum required scope and quota.
3. Review provider audit logs from before the exposure through revocation.
4. Update deployment secrets; never commit the replacement.
5. Record revocation evidence privately.

Repository automation scans for future leaks, but only the credential provider
can prove revocation. Historical credential revocation must therefore be
confirmed by the repository owner before this project is used operationally.

## Operational boundaries

- Use an unfunded public key for quote requests.
- Keep API keys in an environment or secret manager, never TOML or shell
  history.
- Treat quote output as untrusted and non-executable.
- Do not enable archived transaction code.
- Require a new threat model, protocol-specific tests, simulation, bounded loss
  controls, and independent security review before adding execution.
