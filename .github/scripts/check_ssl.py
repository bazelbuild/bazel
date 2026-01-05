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

import ssl
import socket
import datetime
import sys
import yaml
import os
import certifi

# Configuration defaults
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/ssl_domains.yaml')

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {CONFIG_PATH}")
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML config: {exc}")
        sys.exit(1)

def check_domains(domains, warning_days):
    failed_domains = []
    print(f"Checking {len(domains)} domains (Threshold: < {warning_days} days)...\n")
    print(f"{'DOMAIN':<30} | {'DAYS LEFT':<10} | {'STATUS'}")
    print("-" * 60)

    context = ssl.create_default_context(cafile=certifi.where())

    for domain in domains:
        try:
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    expires_str = cert['notAfter']
                    # The format returned by getpeercert() is '%b %d %H:%M:%S %Y GMT'
                    expires_dt = datetime.datetime.strptime(expires_str, '%b %d %H:%M:%S %Y GMT').replace(tzinfo=datetime.timezone.utc)
                    days_left = (expires_dt - datetime.datetime.now(datetime.timezone.utc)).days

                    status = "✅ OK"
                    if days_left < warning_days:
                        status = "❌ EXPIRING SOON"
                        failed_domains.append(f"{domain} ({days_left} days left)")

                    print(f"{domain:<30} | {days_left:<10} | {status}")
        except Exception as e:
            print(f"{domain:<30} | {'ERROR':<10} | 🚨 {str(e)}")
            failed_domains.append(f"{domain} ({str(e)})")

    return failed_domains

if __name__ == "__main__":
    config = load_config()
    domains = config.get('domains', [])
    warning_days = config.get('warning_days', 21)
    
    if not domains:
        print("Error: No domains found in config.")
        sys.exit(1)

    failures = check_domains(domains, warning_days)

    if failures:
        summary = "\n".join([f"- {d}" for d in failures])

        # Write to GITHUB_STEP_SUMMARY only if running in GitHub Actions
        if 'GITHUB_STEP_SUMMARY' in os.environ:
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as fh:
                print("### 🚨 SSL Certificates Expiring Soon", file=fh)
                print(summary, file=fh)

        print("\nCRITICAL ISSUES FOUND:")
        print(summary)
        sys.exit(1)

    print("\nAll certificates look good.")
