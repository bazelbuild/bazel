# Copyright 2026 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to monitor SSL certificate expiration for Bazel domains."""

import datetime
import os
import socket
import ssl
import sys
from typing import Any, Dict, List

import certifi
import yaml

# Constants
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "../config/ssl_domains.yaml"
)
DEFAULT_WARNING_DAYS = 21
HTTPS_PORT = 443
SOCKET_TIMEOUT_SECONDS = 5


class SSLMonitor:
  """Class to manage and check SSL certificate expirations."""

  def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
    self.config_path = config_path
    self.config = self._load_config()
    self.warning_days = self.config.get("warning_days", DEFAULT_WARNING_DAYS)
    self.domains = self.config.get("domains", [])
    self.context = ssl.create_default_context(cafile=certifi.where())

  def _load_config(self) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    if not os.path.exists(self.config_path):
      print(f"Error: Config file not found at {self.config_path}")
      sys.exit(1)

    try:
      with open(self.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        return config if isinstance(config, dict) else {}
    except yaml.YAMLError as exc:
      print(f"Error parsing YAML config: {exc}")
      sys.exit(1)

  def get_days_until_expiration(self, domain: str) -> int:
    """Connects to the domain and retrieves days until certificate expiration."""
    with socket.create_connection(
        (domain, HTTPS_PORT), timeout=SOCKET_TIMEOUT_SECONDS
    ) as sock:
      with self.context.wrap_socket(sock, server_hostname=domain) as ssock:
        cert = ssock.getpeercert()
        if not cert:
          raise ValueError(f"No certificate found for {domain}")

        expires_str = cert["notAfter"]
        # Format: '%b %d %H:%M:%S %Y GMT' (e.g., 'Mar 26 20:13:28 2026 GMT')
        expires_dt = datetime.datetime.strptime(
            expires_str, "%b %d %H:%M:%S %Y GMT"
        ).replace(tzinfo=datetime.timezone.utc)

        delta = expires_dt - datetime.datetime.now(datetime.timezone.utc)
        return delta.days

  def check_all(self) -> List[str]:
    """Checks all configured domains and prints the status."""
    failed_domains = []
    print(
        f"Checking {len(self.domains)} domains (Threshold: <"
        f" {self.warning_days} days)...\n"
    )
    print(f"{'DOMAIN':<35} | {'DAYS LEFT':<10} | {'STATUS'}")
    print("-" * 65)

    for domain in self.domains:
      try:
        days_left = self.get_days_until_expiration(domain)
        status = "✅ OK"
        if days_left < self.warning_days:
          status = "❌ EXPIRING SOON"
          failed_domains.append(f"{domain} ({days_left} days left)")
        print(f"{domain:<35} | {days_left:<10} | {status}")
      except (OSError, ValueError) as e:
        # Catch broad exceptions to ensure we try all domains
        error_msg = str(e)
        print(f"{domain:<35} | {'ERROR':<10} | 🚨 {error_msg}")
        failed_domains.append(f"{domain} ({error_msg})")

    return failed_domains


def report_failures(failures: List[str]):
  """Reports failures to stdout and GitHub Step Summary if available."""
  if not failures:
    print("\nAll certificates look good.")
    return

  summary = "\n".join([f"- {d}" for d in failures])

  # GitHub Action specific summary
  if "GITHUB_STEP_SUMMARY" in os.environ:
    try:
      with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
        fh.write("### 🚨 SSL Certificates Expiring Soon or Failing\n")
        fh.write(summary + "\n")
    except IOError as e:
      print(f"Warning: Could not write to GITHUB_STEP_SUMMARY: {e}")

  print("\nCRITICAL ISSUES FOUND:")
  print(summary)
  sys.exit(1)


def main():

  monitor = SSLMonitor()
  if not monitor.domains:
    print("Error: No domains found in config.")
    sys.exit(1)

  failures = monitor.check_all()
  report_failures(failures)


if __name__ == "__main__":
  main()
