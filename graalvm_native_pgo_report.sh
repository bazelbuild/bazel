#!/usr/bin/env bash
#
# Reproduce the JVM vs GraalVM native-image PGO benchmark report.
#
# Defaults match the latest report in this branch:
#
#   HEAP_SIZE=8g BENCHMARK_COMMIT_COUNT=30 RUNS=3 ./graalvm_native_pgo_report.sh
#
# The script expects benchmark binaries under $BASE/bin by default. Set
# BUILD_BINARIES=1 to rebuild the JVM, native, and native-PGO Bazel binaries
# before running the benchmark.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_HOME="${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}"
BASE="${BASE:-${CACHE_HOME}/graalvm-bazel-native}"
HEAP_SIZE="${HEAP_SIZE:-8g}"
RUNS="${RUNS:-3}"
TARGET="${TARGET:-//src:bazel-bin-dev}"
BENCHMARK_REF="${BENCHMARK_REF:-origin/master}"
BENCHMARK_COMMIT_COUNT="${BENCHMARK_COMMIT_COUNT:-30}"
BUILD_CONFIG="${BUILD_CONFIG:-${REMOTE_CONFIG:-}}"
BUILD_BINARIES="${BUILD_BINARIES:-0}"

RUNNER="${RUNNER:-${REPO_ROOT}/graalvm_native_benchmark_runner.py}"
HYPERFINE="${HYPERFINE:-$(command -v hyperfine || true)}"
if [[ -z "${HYPERFINE}" && -x "${HOME}/.cargo/bin/hyperfine" ]]; then
  HYPERFINE="${HOME}/.cargo/bin/hyperfine"
fi
if [[ -z "${HYPERFINE}" || ! -x "${HYPERFINE}" ]]; then
  echo "hyperfine not found; set HYPERFINE=/path/to/hyperfine" >&2
  exit 1
fi

HEAP_TAG="${HEAP_SIZE//[^[:alnum:]]/}"
ARTIFACTS_ROOT="${BASE}/artifacts"
ARTIFACTS="${ARTIFACTS_ROOT}/pgo_compare_xms${HEAP_TAG}_warmcache_last${BENCHMARK_COMMIT_COUNT}"
REPO_TAG="${BENCHMARK_REF//\//_}"
COMMITS_FILE="${ARTIFACTS_ROOT}/${REPO_TAG}_last${BENCHMARK_COMMIT_COUNT}_commits.txt"
WORKTREE="${WORKTREE:-${BASE}/origin-master-bench}"
DISK_CACHE="${DISK_CACHE:-${BASE}/warmcache_run/disk_cache}"
REPOSITORY_CACHE="${REPOSITORY_CACHE:-${BASE}/warmcache_run/repository_cache}"
BIN_DIR="${BASE}/bin"
JVM_BIN="${JVM_BIN:-${BIN_DIR}/bazel-jvm-pgo-run}"
NATIVE_BIN="${NATIVE_BIN:-${BIN_DIR}/bazel-native-nopgo-run}"
NATIVE_PGO_BIN="${NATIVE_PGO_BIN:-${BIN_DIR}/bazel-native-pgo-run}"
NATIVE_PGO_INSTRUMENTED_BIN="${NATIVE_PGO_INSTRUMENTED_BIN:-${BIN_DIR}/bazel-native-pgo-instrumented-run}"
JVM_OB="${BASE}/output_bases/pgo-compare-xms${HEAP_TAG}-jvm"
NATIVE_OB="${BASE}/output_bases/pgo-compare-xms${HEAP_TAG}-native-nopgo"
NATIVE_PGO_OB="${BASE}/output_bases/pgo-compare-xms${HEAP_TAG}-native-pgo"
NATIVE_PGO_TRAINING_OB="${BASE}/output_bases/pgo-compare-xms${HEAP_TAG}-native-pgo-training"
WARMUP_OB="${BASE}/output_bases/pgo-compare-xms${HEAP_TAG}-cache-warmup"
WARMUP_ARTIFACTS="${ARTIFACTS_ROOT}/pgo_compare_xms${HEAP_TAG}_cache_warmup"
STABLE_REPORT="${ARTIFACTS_ROOT}/graalvm_native_pgo_benchmark_xms${HEAP_TAG}_report.html"
PGO_PROFILE="${PGO_PROFILE:-${ARTIFACTS_ROOT}/bazel_server_native_pgo_profile.iprof}"
PGO_BUILD_INPUT="${REPO_ROOT}/src/bazel_server_native_pgo_profile.iprof"

staged_pgo_profile=0
function cleanup_staged_profile() {
  if [[ "${staged_pgo_profile}" != "1" ]]; then
    return
  fi
  if [[ ! -e "${PGO_BUILD_INPUT}" ]]; then
    staged_pgo_profile=0
    return
  fi
  if cmp -s "${PGO_PROFILE}" "${PGO_BUILD_INPUT}"; then
    rm -f "${PGO_BUILD_INPUT}"
    staged_pgo_profile=0
  else
    echo "Leaving changed PGO build input in place: ${PGO_BUILD_INPUT}" >&2
    staged_pgo_profile=0
  fi
}
trap cleanup_staged_profile EXIT

mkdir -p "${ARTIFACTS_ROOT}" "${BIN_DIR}" "${DISK_CACHE}" "${REPOSITORY_CACHE}"

if ! git -C "${WORKTREE}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "${REPO_ROOT}" worktree add --detach "${WORKTREE}" "${BENCHMARK_REF}"
fi

python3 "${RUNNER}" prepare \
  --worktree "${WORKTREE}"

git -C "${REPO_ROOT}" rev-list --first-parent --max-count="${BENCHMARK_COMMIT_COUNT}" "${BENCHMARK_REF}" \
  | tac > "${COMMITS_FILE}"
BINARY_REVISION="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
BENCHMARK_HEAD="$(git -C "${REPO_ROOT}" rev-parse "${BENCHMARK_REF}")"
BENCHMARK_FIRST="$(head -n 1 "${COMMITS_FILE}")"
BENCHMARK_LAST="$(tail -n 1 "${COMMITS_FILE}")"

build_config_args=()
if [[ -n "${BUILD_CONFIG}" ]]; then
  build_config_args+=("--config=${BUILD_CONFIG}")
fi

if [[ "${BUILD_BINARIES}" == "1" ]]; then
  bazel build \
    "${build_config_args[@]}" \
    --remote_download_outputs=toplevel \
    "--disk_cache=${DISK_CACHE}" \
    //src:bazel-bin-dev \
    //src:bazel-bin_native \
    //src:bazel-bin_native_pgo_instrumented
  rm -f "${JVM_BIN}" "${NATIVE_BIN}" "${NATIVE_PGO_INSTRUMENTED_BIN}"
  cp bazel-bin/src/bazel-dev "${JVM_BIN}"
  cp bazel-bin/src/bazel_native "${NATIVE_BIN}"
  cp bazel-bin/src/bazel_native_pgo_instrumented "${NATIVE_PGO_INSTRUMENTED_BIN}"
  chmod +x "${JVM_BIN}" "${NATIVE_BIN}" "${NATIVE_PGO_INSTRUMENTED_BIN}"

  profile_path="$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${PGO_PROFILE}")"
  build_input_path="$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${PGO_BUILD_INPUT}")"
  if [[ "${profile_path}" == "${build_input_path}" ]]; then
    echo "PGO_PROFILE must be outside the source tree: ${PGO_PROFILE}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${PGO_PROFILE}")"
  rm -f "${PGO_PROFILE}"
  rm -rf "${ARTIFACTS_ROOT}/pgo_training_xms${HEAP_TAG}_last${BENCHMARK_COMMIT_COUNT}"
  python3 "${RUNNER}" prepare \
    --worktree "${WORKTREE}" \
    --commits "${COMMITS_FILE}" \
    --binary "${NATIVE_PGO_INSTRUMENTED_BIN}" \
    --output-base "${NATIVE_PGO_TRAINING_OB}"
  BAZEL_NATIVE_IMAGE_SERVER_ARGS="-XX:ProfilesDumpFile=${PGO_PROFILE}" \
    python3 "${RUNNER}" run \
      --label pgo-training \
      --binary "${NATIVE_PGO_INSTRUMENTED_BIN}" \
      --worktree "${WORKTREE}" \
      --output-base "${NATIVE_PGO_TRAINING_OB}" \
      --commits "${COMMITS_FILE}" \
      --artifacts "${ARTIFACTS_ROOT}/pgo_training_xms${HEAP_TAG}_last${BENCHMARK_COMMIT_COUNT}" \
      --results-csv "${ARTIFACTS_ROOT}/pgo_training_xms${HEAP_TAG}_last${BENCHMARK_COMMIT_COUNT}/results.csv" \
      --disk-cache "${DISK_CACHE}" \
      --repository-cache "${REPOSITORY_CACHE}" \
      --target "${TARGET}"

  # Native Image writes the PGO profile while the instrumented process exits.
  BAZEL_NATIVE_IMAGE_SERVER_ARGS="-XX:ProfilesDumpFile=${PGO_PROFILE}" \
    "${NATIVE_PGO_INSTRUMENTED_BIN}" \
      --output_base="${NATIVE_PGO_TRAINING_OB}" shutdown
  for _ in $(seq 1 30); do
    [[ -s "${PGO_PROFILE}" ]] && break
    sleep 1
  done

  if [[ ! -s "${PGO_PROFILE}" ]]; then
    echo "training did not produce a nonempty PGO profile: ${PGO_PROFILE}" >&2
    exit 1
  fi
  if [[ -e "${PGO_BUILD_INPUT}" ]]; then
    echo "refusing to overwrite existing PGO build input: ${PGO_BUILD_INPUT}" >&2
    exit 1
  fi
  cp "${PGO_PROFILE}" "${PGO_BUILD_INPUT}"
  staged_pgo_profile=1

  bazel build \
    "${build_config_args[@]}" \
    --remote_download_outputs=toplevel \
    "--disk_cache=${DISK_CACHE}" \
    //src:bazel-bin_native_pgo
  rm -f "${NATIVE_PGO_BIN}"
  cp bazel-bin/src/bazel_native_pgo "${NATIVE_PGO_BIN}"
  chmod +x "${NATIVE_PGO_BIN}"
  cleanup_staged_profile
fi

for binary in "${JVM_BIN}" "${NATIVE_BIN}" "${NATIVE_PGO_BIN}"; do
  if [[ ! -x "${binary}" ]]; then
    echo "missing executable benchmark binary: ${binary}" >&2
    echo "Set *_BIN explicitly or rerun with BUILD_BINARIES=1." >&2
    exit 1
  fi
done

echo "Warming disk and repository caches over ${BENCHMARK_REF}."
rm -rf "${WARMUP_ARTIFACTS}" "${WARMUP_OB}"
mkdir -p "${WARMUP_ARTIFACTS}"
python3 "${RUNNER}" prepare \
  --worktree "${WORKTREE}" \
  --commits "${COMMITS_FILE}" \
  --binary "${JVM_BIN}" \
  --output-base "${WARMUP_OB}"
python3 "${RUNNER}" run \
  --label cache-warmup \
  --binary "${JVM_BIN}" \
  --startup-option="--host_jvm_args=-Xms${HEAP_SIZE}" \
  --startup-option="--host_jvm_args=-Xmx${HEAP_SIZE}" \
  --worktree "${WORKTREE}" \
  --output-base "${WARMUP_OB}" \
  --commits "${COMMITS_FILE}" \
  --artifacts "${WARMUP_ARTIFACTS}" \
  --results-csv "${WARMUP_ARTIFACTS}/results.csv" \
  --disk-cache "${DISK_CACHE}" \
  --repository-cache "${REPOSITORY_CACHE}" \
  --target "${TARGET}"
python3 "${RUNNER}" prepare \
  --worktree "${WORKTREE}" \
  --binary "${JVM_BIN}" \
  --output-base "${WARMUP_OB}"
rm -rf "${WARMUP_ARTIFACTS}"

rm -rf "${ARTIFACTS}"
mkdir -p "${ARTIFACTS}"

"${HYPERFINE}" --runs "${RUNS}" --warmup 0 \
  --export-json "${ARTIFACTS}/hyperfine.json" \
  --prepare "python3 ${RUNNER} prepare --worktree ${WORKTREE} --commits ${COMMITS_FILE} --binary ${JVM_BIN} --binary ${NATIVE_BIN} --binary ${NATIVE_PGO_BIN} --output-base ${JVM_OB} --output-base ${NATIVE_OB} --output-base ${NATIVE_PGO_OB}" \
  "python3 ${RUNNER} run --label pgo-jvm --binary ${JVM_BIN} --startup-option=--host_jvm_args=-Xms${HEAP_SIZE} --startup-option=--host_jvm_args=-Xmx${HEAP_SIZE} --worktree ${WORKTREE} --output-base ${JVM_OB} --commits ${COMMITS_FILE} --artifacts ${ARTIFACTS} --results-csv ${ARTIFACTS}/results.csv --disk-cache ${DISK_CACHE} --repository-cache ${REPOSITORY_CACHE} --target ${TARGET}" \
  "env BAZEL_NATIVE_IMAGE_SERVER_XMX=${HEAP_SIZE} BAZEL_NATIVE_IMAGE_SERVER_ARGS=-Xms${HEAP_SIZE} python3 ${RUNNER} run --label pgo-native-nopgo --binary ${NATIVE_BIN} --worktree ${WORKTREE} --output-base ${NATIVE_OB} --commits ${COMMITS_FILE} --artifacts ${ARTIFACTS} --results-csv ${ARTIFACTS}/results.csv --disk-cache ${DISK_CACHE} --repository-cache ${REPOSITORY_CACHE} --target ${TARGET}" \
  "env BAZEL_NATIVE_IMAGE_SERVER_XMX=${HEAP_SIZE} BAZEL_NATIVE_IMAGE_SERVER_ARGS=-Xms${HEAP_SIZE} python3 ${RUNNER} run --label pgo-native-pgo --binary ${NATIVE_PGO_BIN} --worktree ${WORKTREE} --output-base ${NATIVE_PGO_OB} --commits ${COMMITS_FILE} --artifacts ${ARTIFACTS} --results-csv ${ARTIFACTS}/results.csv --disk-cache ${DISK_CACHE} --repository-cache ${REPOSITORY_CACHE} --target ${TARGET}"

report_profile_args=()
if [[ -s "${PGO_PROFILE}" ]]; then
  report_profile_args+=(--pgo-profile "${PGO_PROFILE}")
fi

python3 "${RUNNER}" report \
  --results-csv "${ARTIFACTS}/results.csv" \
  --hyperfine-json "${ARTIFACTS}/hyperfine.json" \
  --output "${ARTIFACTS}/report.html" \
  --title "Bazel GraalVM Native Image PGO Benchmark, ${HEAP_SIZE} Committed Heap" \
  --description "Binary revision: ${BINARY_REVISION}. Workload: ${BENCHMARK_COMMIT_COUNT} sequential first-parent commits ${BENCHMARK_FIRST} through ${BENCHMARK_LAST} from ${BENCHMARK_REF} at ${BENCHMARK_HEAD}, build --nobuild ${TARGET}, caches prewarmed by an unmeasured pass, daemon/output-base cleanup before every Hyperfine repetition. JVM uses --host_jvm_args=-Xms${HEAP_SIZE}/-Xmx${HEAP_SIZE}; native uses BAZEL_NATIVE_IMAGE_SERVER_ARGS=-Xms${HEAP_SIZE} and BAZEL_NATIVE_IMAGE_SERVER_XMX=${HEAP_SIZE}." \
  --note "This run equalizes committed runtime heap at approximately ${HEAP_SIZE}, excludes spawn execution with --nobuild, and preserves per-commit Bazel --profile and --memory_profile artifacts." \
  "${report_profile_args[@]}"

cp "${ARTIFACTS}/report.html" "${STABLE_REPORT}"

python3 - "$ARTIFACTS/results.csv" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open()))
failures = [r for r in rows if r["exit_code"] != "0"]
profiles = [Path(r["profile_path"]) for r in rows]
memory_profiles = [Path(r["memory_profile_path"]) for r in rows]
missing_profiles = [p for p in profiles if not p.exists()]
missing_memory_profiles = [p for p in memory_profiles if not p.exists()]
if failures or missing_profiles or missing_memory_profiles:
    print(f"failures={len(failures)}", file=sys.stderr)
    print(f"missing_profiles={len(missing_profiles)}", file=sys.stderr)
    print(f"missing_memory_profiles={len(missing_memory_profiles)}", file=sys.stderr)
    raise SystemExit(1)
print(f"validated {len(rows)} rows")
print(f"profile files: {len(profiles)}")
print(f"memory profile files: {len(memory_profiles)}")
PY

echo "report: ${ARTIFACTS}/report.html"
echo "stable report: ${STABLE_REPORT}"
