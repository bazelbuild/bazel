#!/bin/bash -eu
#
# Copyright 2016, 2023 The Bazel Authors. All rights reserved.
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
# Tests the behavior of C++ runfiles library.

# --- begin runfiles.bash initialization ---
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
    if [[ -f "$0.runfiles_manifest" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
    elif [[ -f "$0.runfiles/MANIFEST" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
    elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
      export RUNFILES_DIR="$0.runfiles"
    fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_bazel_current_repository_define() {
  cat >> WORKSPACE <<'EOF'
local_repository(
  name = "other_repo",
  path = "other_repo",
)
EOF

  mkdir -p pkg
  cat > pkg/BUILD.bazel <<'EOF'
cc_library(
  name = "library",
  srcs = ["library.cpp"],
  hdrs = ["library.h"],
  visibility = ["//visibility:public"],
)

cc_binary(
  name = "binary",
  srcs = ["binary.cpp"],
  deps = [":library"],
)

cc_test(
  name = "test",
  srcs = ["test.cpp"],
  deps = [":library"],
)
EOF

  cat > pkg/library.cpp <<'EOF'
#include "library.h"
#include <iostream>
void print_repo_name() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
}
EOF

  cat > pkg/library.h <<'EOF'
void print_repo_name();
EOF

  cat > pkg/binary.cpp <<'EOF'
#include <iostream>
#include "library.h"
int main() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
  print_repo_name();
}
EOF

  cat > pkg/test.cpp <<'EOF'
#include <iostream>
#include "library.h"
int main() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
  print_repo_name();
}
EOF

  mkdir -p other_repo
  touch other_repo/WORKSPACE

  mkdir -p other_repo/pkg
  cat > other_repo/pkg/BUILD.bazel <<'EOF'
cc_binary(
  name = "binary",
  srcs = ["binary.cpp"],
  deps = ["@//pkg:library"],
)

cc_test(
  name = "test",
  srcs = ["test.cpp"],
  deps = ["@//pkg:library"],
)
EOF

  cat > other_repo/pkg/binary.cpp <<'EOF'
#include <iostream>
#include "pkg/library.h"
int main() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
  print_repo_name();
}
EOF

  cat > other_repo/pkg/test.cpp <<'EOF'
#include <iostream>
#include "pkg/library.h"
int main() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
  print_repo_name();
}
EOF

  bazel run pkg:binary &>"$TEST_log" || fail "Run should succeed"
  expect_log "in .*pkg.binary.cpp: ''"
  expect_log "in .*pkg.library.cpp: ''"

  bazel test --test_output=streamed pkg:test &>"$TEST_log" || fail "Test should succeed"
  expect_log "in .*pkg.test.cpp: ''"
  expect_log "in .*pkg.library.cpp: ''"

  bazel run @other_repo//pkg:binary &>"$TEST_log" || fail "Run should succeed"
  expect_log "in .*external.other_repo.pkg.binary.cpp: 'other_repo'"
  expect_log "in .*pkg.library.cpp: ''"

  bazel test --test_output=streamed \
    @other_repo//pkg:test &>"$TEST_log" || fail "Test should succeed"
  expect_log "in .*external.other_repo.pkg.test.cpp: 'other_repo'"
  expect_log "in .*pkg.library.cpp: ''"
}

function test_bazel_current_repository_func() {
  cat >> WORKSPACE <<'EOF'
local_repository(
  name = "other_repo",
  path = "other_repo",
)
EOF

  mkdir -p pkg
  cat > pkg/BUILD.bazel <<'EOF'
cc_library(
  name = "library",
  hdrs = ["library.h"],
  deps = ["@bazel_tools//tools/cpp/runfiles", ":gen_library", "@other_repo//gen_pkg:gen_library"],
  visibility = ["//visibility:public"],
)

genrule(
  name = "gen_library_hdr",
  outs = ["gen_library.h"],
  cmd = """
cat >$@ <<EOF_
#include <iostream>

#include "tools/cpp/runfiles/runfiles.h"

using bazel::tools::cpp::runfiles::Runfiles;

inline void gen_print_repo_name() {
  std::cout << "in " << __FILE__ << ": '" << Runfiles::CurrentRepository() << "'" << std::endl;
}
EOF_
"""
)

cc_library(
  name = "gen_library",
  hdrs = ["gen_library.h"],
  deps = ["@bazel_tools//tools/cpp/runfiles"],
)

cc_binary(
  name = "binary",
  srcs = ["binary.cpp"],
  deps = [":library"],
)

cc_test(
  name = "test",
  srcs = ["test.cpp"],
  deps = [":library"],
)

cc_test(
  name = "feature_test",
  srcs = ["feature_test.cpp"],
  deps = ["@bazel_tools//tools/cpp/runfiles"],
)
EOF

  cat > pkg/library.cpp <<'EOF'
#include "library.h"
#include <iostream>
void print_repo_name() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
}
EOF

  cat > pkg/library.h <<'EOF'
#include <iostream>
#include "tools/cpp/runfiles/runfiles.h"
#include "pkg/gen_library.h"
#include "gen_pkg/gen_library.h"

using bazel::tools::cpp::runfiles::Runfiles;

inline void print_repo_name() {
  std::cout << "in " << __FILE__ << ": '" << Runfiles::CurrentRepository() << "'" << std::endl;
  gen_print_repo_name();
  other_repo_gen_print_repo_name();
}
EOF

  cat > pkg/binary.cpp <<'EOF'
#include <iostream>
#include "library.h"
int main() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
  print_repo_name();
}
EOF

  cat > pkg/test.cpp <<'EOF'
#include <iostream>
#include "library.h"
int main() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
  print_repo_name();
}
EOF

  cat > pkg/feature_test.cpp <<'EOF'
#include "tools/cpp/runfiles/runfiles.h"

int main() {
#if defined(BAZEL_TOOLS_CPP_RUNFILES_HAS_BUILTIN_FILE)
  return 0;
#else
  return 1;
#endif
}
EOF

  mkdir -p other_repo
  touch other_repo/WORKSPACE

  mkdir -p other_repo/pkg
  cat > other_repo/pkg/BUILD.bazel <<'EOF'
cc_binary(
  name = "binary",
  srcs = ["binary.cpp"],
  deps = ["@//pkg:library"],
)

cc_test(
  name = "test",
  srcs = ["test.cpp"],
  deps = ["@//pkg:library"],
)
EOF

  cat > other_repo/pkg/binary.cpp <<'EOF'
#include <iostream>
#include "pkg/library.h"
int main() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
  print_repo_name();
}
EOF

  cat > other_repo/pkg/test.cpp <<'EOF'
#include <iostream>
#include "pkg/library.h"
int main() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
  print_repo_name();
}
EOF

  mkdir -p other_repo/gen_pkg
  cat > other_repo/gen_pkg/BUILD.bazel <<'EOF'
genrule(
  name = "gen_library_hdr",
  outs = ["gen_library.h"],
  cmd = """
cat >$@ <<EOF_
#include <iostream>

#include "tools/cpp/runfiles/runfiles.h"

using bazel::tools::cpp::runfiles::Runfiles;

inline void other_repo_gen_print_repo_name() {
  std::cout << "in " << __FILE__ << ": '" << Runfiles::CurrentRepository() << "'" << std::endl;
}
EOF_
"""
)

cc_library(
  name = "gen_library",
  hdrs = ["gen_library.h"],
  deps = ["@bazel_tools//tools/cpp/runfiles"],
  visibility = ["//visibility:public"],
)
EOF

  bazel build pkg:feature_test &>"$TEST_log" || fail "Build should succeed"
  if ! bazel test pkg:feature_test &>"$TEST_log"; then
    echo "test skipped because __builtin_FILE is not supported" >&2
    return
  fi

  bazel run pkg:binary &>"$TEST_log" || fail "Run should succeed"
  expect_log "in .*pkg.binary.cpp: ''"
  expect_log "in .*pkg.library.h: ''"
  expect_log "in .*pkg.gen_library.h: ''"
  expect_log "in .*gen_pkg.gen_library.h: 'other_repo'"

  bazel test --test_output=streamed pkg:test &>"$TEST_log" || fail "Test should succeed"
  expect_log "in .*pkg.test.cpp: ''"
  expect_log "in .*pkg.library.h: ''"
  expect_log "in .*pkg.gen_library.h: ''"
  expect_log "in .*gen_pkg.gen_library.h: 'other_repo'"

  bazel run @other_repo//pkg:binary &>"$TEST_log" || fail "Run should succeed"
  expect_log "in .*external.other_repo/pkg/binary.cpp: 'other_repo'"
  expect_log "in .*pkg.library.h: ''"
  expect_log "in .*pkg.gen_library.h: ''"
  expect_log "in .*gen_pkg.gen_library.h: 'other_repo'"

  bazel test --test_output=streamed \
    @other_repo//pkg:test &>"$TEST_log" || fail "Test should succeed"
  expect_log "in .*external.other_repo.pkg.test.cpp: 'other_repo'"
  expect_log "in .*pkg.library.h: ''"
  expect_log "in .*pkg.gen_library.h: ''"
  expect_log "in .*gen_pkg.gen_library.h: 'other_repo'"
}

run_suite "cc_runfiles_test"
