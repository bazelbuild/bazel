#!/usr/bin/env bash
# SECURITY-SCANNER TEST FIXTURE - DO NOT MERGE
#
# This script is intentionally added on a fork branch, in isolation, to
# verify that our SAST / PR security-gate tooling flags known vulnerability
# classes before a real change could reach this repo. It is NOT wired up to
# action.yml or any workflow, so it never executes as part of the action.
# It must never be merged into any branch that ships.
#
# Intentional vulnerabilities:
#   1. CWE-798 Hardcoded Credentials (severity: High)
#      Detected by: Gitleaks, TruffleHog, GitHub secret scanning,
#      Semgrep generic.secrets.security.detected-aws-credentials
#   2. CWE-78  OS Command Injection (severity: Critical)
#      Detected by: Bandit B602/B605-equivalent shell checks, Semgrep
#      bash.lang.security.ssh-injection / dangerous-eval,
#      CodeQL js/shell-command-injection-from-environment (bash analogue)

set -euo pipefail

# --- Vulnerability 1: hardcoded credential (High) ---
AWS_SECRET_ACCESS_KEY="AKIAIOSFODNN7EXAMPLEDEMOKEY123"

# --- Vulnerability 2: command injection via unsanitized input (Critical) ---
run_user_supplied_command() {
    local user_input="$1"
    # VULNERABLE: untrusted input passed straight to eval instead of being
    # treated as data (e.g. via an array, or validated against an allowlist).
    eval "$user_input"
}
