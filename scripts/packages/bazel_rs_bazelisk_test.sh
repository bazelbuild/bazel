#!/usr/bin/env bash
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

# Integration test verifying that `package_bazel_rs.sh` produces artifacts
# compatible with `scripts/packages/bazel.sh` and `bazelisk`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TEST_TMP="$(mktemp -d "${TMPDIR:-/tmp}/bazel-rs-pkg-test.XXXXXX")"
trap 'rm -rf "${TEST_TMP}"' EXIT

READ_VERSION="$(cat "${WORKSPACE_ROOT}/.bazelversion" 2>/dev/null || echo "9.3.1")"
REL_DIR="${TEST_TMP}/release"
OB_DIR="${TEST_TMP}/ob"

# 1. Package bazel-rs using the dev profile for fast test execution
"${SCRIPT_DIR}/package_bazel_rs.sh" \
  --version="${READ_VERSION}" \
  --output_dir="${REL_DIR}" \
  --profile=dev

OS_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH_NAME="$(uname -m)"
case "${ARCH_NAME}" in
  x86_64|amd64) BAZELISK_ARCH="x86_64" ;;
  aarch64|arm64) BAZELISK_ARCH="arm64" ;;
  *) BAZELISK_ARCH="x86_64" ;;
esac

PACKAGED_BIN="${REL_DIR}/bazel-${READ_VERSION}-${OS_NAME}-${BAZELISK_ARCH}"
PACKAGED_SHA="${PACKAGED_BIN}.sha256"

[[ -x "${PACKAGED_BIN}" ]] || { echo "FAIL: Missing executable ${PACKAGED_BIN}" >&2; exit 1; }
[[ -f "${PACKAGED_SHA}" ]] || { echo "FAIL: Missing sha256 ${PACKAGED_SHA}" >&2; exit 1; }

# 2. Verify `--version` output format required by Bazelisk and `scripts/packages/bazel.sh`
VER_OUT="$(BAZEL_EMBED_LABEL="${READ_VERSION}" "${PACKAGED_BIN}" --version)"
if [[ "${VER_OUT}" != "bazel ${READ_VERSION}" ]]; then
  echo "FAIL: Expected 'bazel ${READ_VERSION}', got '${VER_OUT}'" >&2
  exit 1
fi

# 3. Verify `scripts/packages/bazel.sh` wrapper discovers and runs the packaged binary
WRAPPER_DIR="${TEST_TMP}/wrapper_bin"
mkdir -p "${WRAPPER_DIR}"
cp "${SCRIPT_DIR}/bazel.sh" "${WRAPPER_DIR}/bazel"
chmod +x "${WRAPPER_DIR}/bazel"
# bazel.sh uses uname -m directly for os_arch_suffix
cp "${PACKAGED_BIN}" "${WRAPPER_DIR}/bazel-${READ_VERSION}-${OS_NAME}-${ARCH_NAME}"
chmod +x "${WRAPPER_DIR}/bazel-${READ_VERSION}-${OS_NAME}-${ARCH_NAME}"

WRAPPER_VER="$(cd "${WORKSPACE_ROOT}" && "${WRAPPER_DIR}/bazel" --version)"
if [[ "${WRAPPER_VER}" != "bazel ${READ_VERSION}" ]]; then
  echo "FAIL: Wrapper returned '${WRAPPER_VER}', expected 'bazel ${READ_VERSION}'" >&2
  exit 1
fi

# 4. If bazelisk is installed as `bazel`, verify `USE_BAZEL_VERSION=<path>` works end-to-end
if command -v bazel &>/dev/null; then
  BAZELISK_OUT="$(cd "${WORKSPACE_ROOT}" && BAZEL_EMBED_LABEL="${READ_VERSION}" USE_BAZEL_VERSION="${PACKAGED_BIN}" bazel --output_base="${OB_DIR}" info release)"
  if [[ "${BAZELISK_OUT}" != *"bazel-rs"* ]]; then
    echo "FAIL: Bazelisk did not invoke bazel-rs: ${BAZELISK_OUT}" >&2
    exit 1
  fi
fi

echo "PASS: bazel-rs packaging, wrapper, and Bazelisk compatibility verified."

