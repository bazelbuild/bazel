#!/usr/bin/env bash
#
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
#
# Test that Windows Defender is not running on Windows.

set -euo pipefail

# Check if running on Windows.
case "$(uname -s 2>/dev/null | tr '[:upper:]' '[:lower:]')" in
  msys*|mingw*|cygwin*)
    is_windows=1
    ;;
  *)
    is_windows=0
    ;;
esac

if [[ "$is_windows" -eq 0 ]]; then
  echo "Not running on Windows; test passed."
  exit 0
fi

failed=0

# Check if the WinDefend service is running
if command -v sc.exe >/dev/null 2>&1; then
  if sc.exe query WinDefend 2>/dev/null | grep -q "RUNNING"; then
    echo "ERROR: Windows Defender service (WinDefend) is running." >&2
    failed=1
  fi
fi

# Check if the MsMpEng.exe scanning process is running
if command -v tasklist.exe >/dev/null 2>&1; then
  if tasklist.exe /FI "IMAGENAME eq MsMpEng.exe" 2>/dev/null | grep -iq "MsMpEng.exe"; then
    echo "ERROR: Windows Defender process (MsMpEng.exe) is running." >&2
    failed=1
  fi
fi

# As a fallback, check via PowerShell if available
if [[ "$failed" -eq 0 ]] && command -v powershell.exe >/dev/null 2>&1; then
  defender_running="$(powershell.exe -NoProfile -Command "
    \$proc = Get-Process -Name MsMpEng -ErrorAction SilentlyContinue
    \$svc = Get-Service -Name WinDefend -ErrorAction SilentlyContinue | Where-Object { \$_.Status -eq 'Running' }
    if (\$proc -ne \$null -or \$svc -ne \$null) { 'RUNNING' } else { 'STOPPED' }
  " 2>/dev/null | tr -d '\r\n')"
  if [[ "$defender_running" == *"RUNNING"* ]]; then
    echo "ERROR: Windows Defender detected via PowerShell." >&2
    failed=1
  fi
fi

if [[ "$failed" -ne 0 ]]; then
  echo "FAIL: Windows Defender is active on this machine! It must be disabled to prevent severe test slowdowns." >&2
  exit 1
fi

echo "Windows Defender is not running; test passed."
exit 0
