#!/bin/bash
#
# Copyright 2024 The Bazel Authors. All rights reserved.
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
# bash_completion_test.sh: tests of bash command completion.

# --- begin runfiles.bash initialization v3 ---
# Copy-pasted from the Bazel Bash runfiles library v3.
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

: ${DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
source ${DIR}/testenv.sh || { echo "testenv.sh not found!" >&2; exit 1; }

JQ_SCRIPT_FILE="$(rlocation io_bazel/scripts/bazel-lockfile-merge.jq)"
JQ="$(rlocation $JQ_RLOCATIONPATH)"

function do_merge() {
  local base="$1"
  local left="$2"
  local right="$3"

  assert_not_contains "'" "$JQ_SCRIPT_FILE"
  # Simulate the setup of a git merge driver, which can only be configured as a
  # single command passed to sh and overwrites the "left" version. The check
  # above verifies that wrapping the jq script in single quotes is sufficient to
  # escape it here.
  jq_script="$(cat "$JQ_SCRIPT_FILE")"
  merge_cmd="\"$JQ\" -s '${jq_script}' -- $base $left $right > ${left}.jq_tmp && mv ${left}.jq_tmp ${left}"
  sh -c "${merge_cmd}" || fail "merge failed"
}

function test_synthetic_merge() {
  cat > base <<'EOF'
{
  "lockFileVersion": 10,
  "registryFileHashes": {
    "https://example.org/modules/bar/0.9/MODULE.bazel": "1234",
    "https://example.org/modules/foo/1.0/MODULE.bazel": "1234"
  },
  "selectedYankedVersions": {},
  "moduleExtensions": {
    "//:rbe_extensions.bzl%bazel_rbe_deps": {
      "general": {
        "bzlTransitiveDigest": "3Qxu4ylcYD3RTWLhk5k/59p/CwZ4tLdSgYnmBXYgAtc=",
        "recordedFileInputs": {},
        "recordedDirentsInputs": {},
        "envVariables": {},
        "generatedRepoSpecs": {
          "rbe_ubuntu2004": {
            "bzlFile": "@@+bazel_test_deps+bazelci_rules//:rbe_repo.bzl",
            "ruleClassName": "rbe_preconfig",
            "attributes": {
              "toolchain": "ubuntu2004"
            }
          }
        },
        "recordedRepoMappingEntries": [
          [
            "",
            "bazelci_rules",
            "+bazel_test_deps+bazelci_rules"
          ]
        ]
      }
    },
    "@@rules_python+//python/extensions:python.bzl%python": {
      "general": {
        "repo1": "old_args"
      }
    }
  }
}
EOF
  cat > left <<'EOF'
{
  "lockFileVersion": 10,
  "registryFileHashes": {
    "https://example.org/modules/bar/0.9/MODULE.bazel": "1234",
    "https://example.org/modules/baz/2.0/MODULE.bazel": "1234",
    "https://example.org/modules/foo/1.0/MODULE.bazel": "1234"
  },
  "selectedYankedVersions": {
    "bbb@1.0": "also dubious"
  },
  "moduleExtensions": {
    "@@rules_python+//python/extensions:python.bzl%python": {
      "general": {
        "repo1": "new_args"
      }
    },
    "@@rules_python+//python/extensions/private:internal_deps.bzl%internal_deps": {
      "os:linux,arch:aarch64": {
        "repo2": "aarch64_args"
      }
    }
  }
}
EOF
  cat > right <<'EOF'
{
  "lockFileVersion": 10,
  "registryFileHashes": {
    "https://example.org/modules/bar/0.9/MODULE.bazel": "1234",
    "https://example.org/modules/bar/1.0/MODULE.bazel": "1234",
    "https://example.org/modules/foo/1.0/MODULE.bazel": "1234"
  },
  "selectedYankedVersions": {
    "aaa@1.0": "dubious"
  },
  "moduleExtensions": {
    "//:rbe_extensions.bzl%bazel_rbe_deps": {
      "general": {
        "bzlTransitiveDigest": "changed",
        "recordedFileInputs": {},
        "recordedDirentsInputs": {},
        "envVariables": {},
        "generatedRepoSpecs": {
          "rbe_ubuntu2004": {
            "bzlFile": "@@+bazel_test_deps+bazelci_rules//:rbe_repo.bzl",
            "ruleClassName": "rbe_preconfig",
            "attributes": {
              "toolchain": "ubuntu2004"
            }
          }
        },
        "recordedRepoMappingEntries": [
          [
            "",
            "bazelci_rules",
            "+bazel_test_deps+bazelci_rules"
          ]
        ]
      }
    },
    "@@rules_python+//python/extensions:python.bzl%python": {
      "general": {
        "repo1": "old_args"
      }
    },
    "@@rules_python+//python/extensions/private:internal_deps.bzl%internal_deps": {
      "os:linux,arch:amd64": {
        "repo2": "amd64_args"
      }
    }
  }
}
EOF
  cat > expected <<'EOF'
{
  "lockFileVersion": 10,
  "registryFileHashes": {
    "https://example.org/modules/bar/0.9/MODULE.bazel": "1234",
    "https://example.org/modules/bar/1.0/MODULE.bazel": "1234",
    "https://example.org/modules/baz/2.0/MODULE.bazel": "1234",
    "https://example.org/modules/foo/1.0/MODULE.bazel": "1234"
  },
  "selectedYankedVersions": {
    "aaa@1.0": "dubious",
    "bbb@1.0": "also dubious"
  },
  "moduleExtensions": {
    "//:rbe_extensions.bzl%bazel_rbe_deps": {
      "general": {
        "bzlTransitiveDigest": "changed",
        "recordedFileInputs": {},
        "recordedDirentsInputs": {},
        "envVariables": {},
        "generatedRepoSpecs": {
          "rbe_ubuntu2004": {
            "bzlFile": "@@+bazel_test_deps+bazelci_rules//:rbe_repo.bzl",
            "ruleClassName": "rbe_preconfig",
            "attributes": {
              "toolchain": "ubuntu2004"
            }
          }
        },
        "recordedRepoMappingEntries": [
          [
            "",
            "bazelci_rules",
            "+bazel_test_deps+bazelci_rules"
          ]
        ]
      }
    },
    "@@rules_python+//python/extensions:python.bzl%python": {
      "general": {
        "repo1": "new_args"
      }
    },
    "@@rules_python+//python/extensions/private:internal_deps.bzl%internal_deps": {
      "os:linux,arch:aarch64": {
        "repo2": "aarch64_args"
      },
      "os:linux,arch:amd64": {
        "repo2": "amd64_args"
      }
    }
  }
}
EOF

  do_merge base left right
  diff -u expected left || fail "output differs"
}

function test_complex_identity_merge() {
  test_lockfile="$(rlocation io_bazel/src/test/tools/bzlmod/MODULE.bazel.lock)"
  cp "$test_lockfile" base
  cp "$test_lockfile" left
  cp "$test_lockfile" right

  do_merge base left right
  diff -u $test_lockfile left || fail "output differs"
}

function test_merge_across_versions() {
  test_lockfile="$(rlocation io_bazel/src/test/tools/bzlmod/MODULE.bazel.lock)"
  cp "$test_lockfile" base
  cp "$test_lockfile" left
  cat > right <<'EOF'
{
  "lockFileVersion": 9,
  "weirdField": {}
}
EOF

  do_merge base left right
  diff -u $test_lockfile left || fail "output differs"
}

function test_outdated_versions_only() {
  cat > base <<'EOF'
{
  "lockFileVersion": 9,
  "weirdField": {}
}
EOF
  cat > left <<'EOF'
{
  "lockFileVersion": 8
}
EOF
  cat > right <<'EOF'
{
  "lockFileVersion": 7
}
EOF

  do_merge base left right
  diff -u base left || fail "output differs"
}

run_suite "Tests of bash completion of 'blaze' command."
