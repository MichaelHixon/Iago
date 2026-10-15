# Security policy

## Reporting a vulnerability in Iago

Please report it privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability**. That opens a private advisory only the maintainer can see. Do not open a public issue for a vulnerability.

Include what you found, how to reproduce it, and what an attacker could do with it.

## What counts

- A way for Iago to send traffic to a target the operator did not choose, or to a non-local target without `--authorized`.
- A way for a crafted attack file, scenario file, artifact or replay fixture to run code, write outside the output directories, or reach the network.
- A report or transcript that leaks a secret Iago was told to keep out of it.

## What does not

A technique in the attack library that bypasses some model's guardrail is the tool working as designed, not a vulnerability in Iago. If it bypasses a model you do not maintain, report it to that model's provider.

If you found a working bypass of a model through Iago, please do not publish it in an issue or pull request before that model's provider has had a chance to fix it.

## Supported versions

Only the latest commit on `main` is supported. Iago has one maintainer, so responses are best effort.
