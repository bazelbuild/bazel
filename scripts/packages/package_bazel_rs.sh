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

# Builds and packages the Rust-based Bazel binary (`bazel-rs`) using the exact
# artifact naming and SHA-256 layout expected by Bazelisk, GitHub Releases, and
# https://releases.bazel.build/<version>/release/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

detect_os() {
  case "$(uname -s | tr '[:upper:]' '[:lower:]')" in
    linux*) echo "linux" ;;
    darwin*) echo "darwin" ;;
    msys*|mingw*|cygwin*|windows*) echo "windows" ;;
    *) echo "linux" ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64) echo "x86_64" ;;
    aarch64|arm64) echo "arm64" ;;
    *) echo "x86_64" ;;
  esac
}

 compute_sha256() {
  local file="$1"
  if command -v sha256sum &>/dev/null; then
    (cd "$(dirname "${file}")" && sha256sum "$(basename "${file}")")
  else
    (cd "$(dirname "${file}")" && shasum -a 256 "$(basename "${file}")")
  fi
}

VERSION="${BAZEL_VERSION:-}"
TARGET_OS="$(detect_os)"
TARGET_ARCH="$(detect_arch)"
OUTPUT_DIR="${WORKSPACE_ROOT}/output/release"
PREBUILT_BINARY=""
CARGO_PROFILE="release"

for arg in "$@"; do
  case "${arg}" in
    --version=*) VERSION="${arg#*=}" ;;
    --os=*) TARGET_OS="${arg#*=}" ;;
    --arch=*) TARGET_ARCH="${arg#*=}" ;;
    --output_dir=*) OUTPUT_DIR="${arg#*=}" ;;
    --binary=*) PREBUILT_BINARY="${arg#*=}" ;;
    --profile=*) CARGO_PROFILE="${arg#*=}" ;;
    *)
      echo "Unknown option: ${arg}" >&2
      echo "Usage: $0 [--version=VER] [--os=OS] [--arch=ARCH] [--output_dir=DIR] [--binary=PATH] [--profile=release|dev]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${VERSION}" ]]; then
  if [[ -f "${WORKSPACE_ROOT}/.bazelversion" ]]; then
    read -r VERSION < "${WORKSPACE_ROOT}/.bazelversion" || true
  fi
  VERSION="${VERSION:-10.0.0-prerelease}"
fi

EXE_EXT=""
if [[ "${TARGET_OS}" == "windows" ]]; then
  EXE_EXT=".exe"
fi

if [[ -z "${PREBUILT_BINARY}" ]]; then
  echo "+++ Building bazel-rs (version=${VERSION}, os=${TARGET_OS}, arch=${TARGET_ARCH}, profile=${CARGO_PROFILE})"
  (
    cd "${WORKSPACE_ROOT}"
    if [[ "${CARGO_PROFILE}" == "release" ]]; then
      BAZEL_EMBED_LABEL="${VERSION}" cargo build --release --bin bazel-rs
      PREBUILT_BINARY="${WORKSPACE_ROOT}/target/release/bazel-rs${EXE_EXT}"
    else
      BAZEL_EMBED_LABEL="${VERSION}" cargo build --bin bazel-rs
      PREBUILT_BINARY="${WORKSPACE_ROOT}/target/debug/bazel-rs${EXE_EXT}"
    fi
  )
  if [[ "${CARGO_PROFILE}" == "release" ]]; then
    PREBUILT_BINARY="${WORKSPACE_ROOT}/target/release/bazel-rs${EXE_EXT}"
  else
    PREBUILT_BINARY="${WORKSPACE_ROOT}/target/debug/bazel-rs${EXE_EXT}"
  fi
fi

if [[ ! -f "${PREBUILT_BINARY}" ]]; then
  echo "ERROR: Built binary not found at ${PREBUILT_BINARY}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

BAZELISK_BINARY_NAME="bazel-${VERSION}-${TARGET_OS}-${TARGET_ARCH}${EXE_EXT}"
BAZELISK_NOJDK_NAME="bazel_nojdk-${VERSION}-${TARGET_OS}-${TARGET_ARCH}${EXE_EXT}"

cp -f "${PREBUILT_BINARY}" "${OUTPUT_DIR}/${BAZELISK_BINARY_NAME}"
chmod 0755 "${OUTPUT_DIR}/${BAZELISK_BINARY_NAME}"
compute_sha256 "${OUTPUT_DIR}/${BAZELISK_BINARY_NAME}" > "${OUTPUT_DIR}/${BAZELISK_BINARY_NAME}.sha256"

cp -f "${PREBUILT_BINARY}" "${OUTPUT_DIR}/${BAZELISK_NOJDK_NAME}"
chmod 0755 "${OUTPUT_DIR}/${BAZELISK_NOJDK_NAME}"
compute_sha256 "${OUTPUT_DIR}/${BAZELISK_NOJDK_NAME}" > "${OUTPUT_DIR}/${BAZELISK_NOJDK_NAME}.sha256"

echo "+++ Packaged Bazelisk release artifacts in ${OUTPUT_DIR}:"
echo "  - ${OUTPUT_DIR}/${BAZELISK_BINARY_NAME}"
echo "  - ${OUTPUT_DIR}/${BAZELISK_BINARY_NAME}.sha256"
echo "  - ${OUTPUT_DIR}/${BAZELISK_NOJDK_NAME}"
echo "  - ${OUTPUT_DIR}/${BAZELISK_NOJDK_NAME}.sha256"

