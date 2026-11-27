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

# Compiles first-party server Rust FFI crates listed in a generated manifest
# under derived/rust_bootstrap/<platform>.sh into a shared library (`cdylib`).

set -euo pipefail

PLATFORM="$1"
OUT="$2"

MANIFEST_DIR="${RUST_BOOTSTRAP_DIR:-derived/rust_bootstrap}"
MANIFEST="${MANIFEST_DIR}/${PLATFORM}.sh"
# macOS uses darwin.sh; Unix platforms without a dedicated manifest (e.g.
# FreeBSD/OpenBSD) fall back to linux.sh.
if [[ ! -f "${MANIFEST}" && "${PLATFORM}" != "windows" && -f "${MANIFEST_DIR}/linux.sh" ]]; then
  MANIFEST="${MANIFEST_DIR}/linux.sh"
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo >&2 "ERROR: Missing ${MANIFEST}. Stage 1 bootstrapping requires bazel-distfile."
  exit 1
fi

TMPDIR_RS="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_RS}"' EXIT

if [[ "${PLATFORM}" == "windows" ]]; then
  TMPDIR_OUT="$(cygpath -m "${TMPDIR_RS}")"
  OUT="$(cygpath -m "${OUT}")"
  CODEGEN_FLAGS=(-C target-feature=+crt-static)
else
  TMPDIR_OUT="${TMPDIR_RS}"
  CODEGEN_FLAGS=(-C relocation-model=pic)
fi

RUSTC_BIN=""
RUSTC_SYSROOT_ARGS=()
UMBRELLA_EXTERNS=()
UMBRELLA_COUNT=0
: > "${TMPDIR_RS}/umbrella.rs"

ensure_rustc() {
  if [[ -n "${RUSTC_BIN}" ]]; then
    return
  fi
  RUSTC_BIN="$(command -v "${RUSTC:-rustc}" || true)"
  if [[ -z "${RUSTC_BIN}" ]]; then
    echo >&2 "ERROR: Could not find '${RUSTC:-rustc}'. Install rustc or set RUSTC=/path/to/rustc."
    exit 1
  fi
  local host candidate
  host="$("${RUSTC_BIN}" -vV | sed -n 's/^host: //p' | tr -d '\r')"
  candidate="$(dirname "$(dirname "${RUSTC_BIN}")")"
  if ls "${candidate}/lib/rustlib/${host}/lib/"*.rlib >/dev/null 2>&1; then
    if [[ "${PLATFORM}" == "windows" ]]; then
      candidate="$(cygpath -m "${candidate}")"
    fi
    RUSTC_SYSROOT_ARGS=(--sysroot "${candidate}")
  fi
}

rust_crate() {
  ensure_rustc
  local cid="$1" name="$2" edition="$3" root="$4"
  shift 4
  local env_args=() extra_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --env)
        env_args+=("$2")
        shift 2
        ;;
      --extern)
        local spec="$2"
        extra_args+=(--extern "${spec%%=*}=${TMPDIR_OUT}/${spec#*=}")
        shift 2
        ;;
      --)
        shift
        extra_args+=("$@")
        break
        ;;
      *)
        echo >&2 "ERROR: Unknown rust_crate argument: $1"
        exit 1
        ;;
    esac
  done

  local rlib="${TMPDIR_OUT}/lib${name}-${cid}.rlib"
  local cmd=(
    "${RUSTC_BIN}"
    ${RUSTC_SYSROOT_ARGS[@]+"${RUSTC_SYSROOT_ARGS[@]}"}
    --crate-name="${name}"
    --edition="${edition}"
    --crate-type=rlib
    -C opt-level=2
    -C embed-bitcode=no
    "${CODEGEN_FLAGS[@]}"
    -C metadata="${cid}"
    -L "dependency=${TMPDIR_OUT}"
    ${extra_args[@]+"${extra_args[@]}"}
    "${root}"
    -o "${rlib}"
  )
  if [[ ${#env_args[@]} -gt 0 ]]; then
    env "${env_args[@]}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi

  local alias="_crate_${UMBRELLA_COUNT}"
  UMBRELLA_COUNT=$((UMBRELLA_COUNT + 1))
  printf 'extern crate %s as _;\n' "${alias}" >> "${TMPDIR_RS}/umbrella.rs"
  UMBRELLA_EXTERNS+=(--extern "${alias}=${rlib}")
}

# shellcheck source=/dev/null
source "${MANIFEST}"

if [[ ${UMBRELLA_COUNT} -eq 0 ]]; then
  exit 0
fi

"${RUSTC_BIN}" \
  ${RUSTC_SYSROOT_ARGS[@]+"${RUSTC_SYSROOT_ARGS[@]}"} \
  --edition=2024 \
  --crate-type=cdylib \
  --crate-name=rust_ffi_umbrella \
  -C opt-level=2 \
  -C embed-bitcode=no \
  "${CODEGEN_FLAGS[@]}" \
  -L "dependency=${TMPDIR_OUT}" \
  "${UMBRELLA_EXTERNS[@]}" \
  "${TMPDIR_OUT}/umbrella.rs" \
  -o "${OUT}"
