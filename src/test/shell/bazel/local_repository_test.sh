#!/bin/bash
#
# Copyright 2015 Google Inc. All rights reserved.
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
# Test the local_repository binding
#

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

# Uses a glob from a different repository for a runfile.
# This create two repositories and populate them with basic build files:
#
# ${WORKSPACE_DIR}/
#     WORKSPACE
#     zoo/
#       BUILD
#       dumper.sh
#     red/
#       BUILD
#       day-keeper
# repo2/
#   red/
#     BUILD
#     baby-panda
#
# dumper.sh should be able to dump the contents of baby-panda.
function test_globbing_external_directory() {
  create_new_workspace
  repo2=${new_workspace_dir}

  mkdir -p red
  cat > red/BUILD <<EOF
filegroup(
    name = "panda",
    srcs = glob(['*-panda']),
    visibility = ["//visibility:public"],
)
EOF

  echo "rawr" > red/baby-panda

  cd ${WORKSPACE_DIR}
  mkdir -p {zoo,red}
  cat > WORKSPACE <<EOF
local_repository(name = 'pandas', path = '${repo2}')
bind(name = 'red', actual = '@pandas//red:panda')
EOF

  cat > zoo/BUILD <<EOF
sh_binary(
    name = "dumper",
    srcs = ["dumper.sh"],
    data = ["//external:red", "//red:keepers"]
)
EOF

  cat > zoo/dumper.sh <<EOF
#!/bin/bash
cat external/pandas/red/baby-panda
cat red/day-keeper
EOF
  chmod +x zoo/dumper.sh

  cat > red/BUILD <<EOF
filegroup(
    name = "keepers",
    srcs = glob(['*-keeper']),
    visibility = ["//visibility:public"],
)
EOF

  echo "feed bamboo" > red/day-keeper


  bazel run //zoo:dumper >& $TEST_log || fail "Failed to build/run zoo"
  expect_log "rawr" "//external runfile not cat-ed"
  expect_log "feed bamboo" \
    "runfile in the same package as //external runfiles not cat-ed"
}

# Tests using a Java dependency.
function test_local_repository_java() {
  create_new_workspace
  repo2=$new_workspace_dir

  mkdir -p carnivore
  cat > carnivore/BUILD <<EOF
java_library(
    name = "mongoose",
    srcs = ["Mongoose.java"],
    visibility = ["//visibility:public"],
)
EOF
  cat > carnivore/Mongoose.java <<EOF
package carnivore;
public class Mongoose {
    public static void frolic() {
        System.out.println("Tra-la!");
    }
}
EOF

  cd ${WORKSPACE_DIR}
  cat > WORKSPACE <<EOF
local_repository(name = 'endangered', path = '$repo2')
bind(name = 'mongoose', actual = '@endangered//carnivore:mongoose')
EOF

  mkdir -p zoo
  cat > zoo/BUILD <<EOF
java_binary(
    name = "ball-pit",
    srcs = ["BallPit.java"],
    main_class = "BallPit",
    deps = ["//external:mongoose"],
)
EOF

  cat > zoo/BallPit.java <<EOF
import carnivore.Mongoose;

public class BallPit {
    public static void main(String args[]) {
        Mongoose.frolic();
    }
}
EOF

  bazel run //zoo:ball-pit >& $TEST_log
  expect_log "Tra-la!"
}

function test_new_local_repository() {
  bazel clean

  # Create a non-Bazel directory.
  project_dir=$TEST_TMPDIR/project
  mkdir -p $project_dir
  outside_dir=$TEST_TMPDIR/outside
  mkdir -p $outside_dir
  package_dir=$project_dir/carnivore
  mkdir $package_dir
  # Be tricky with absolute symlinks to make sure that Bazel still acts as
  # though external repositories are immutable.
  ln -s $outside_dir/Mongoose.java $package_dir/Mongoose.java

  cat > $package_dir/Mongoose.java <<EOF
package carnivore;
public class Mongoose {
    public static void frolic() {
        System.out.println("Tra-la!");
    }
}
EOF

  build_file=BUILD.carnivore
  cat > WORKSPACE <<EOF
new_local_repository(
    name = 'endangered',
    path = '$project_dir',
    build_file = '$build_file',
)
bind(
    name = 'mongoose',
    actual = '@endangered//:mongoose'
)
EOF

   mkdir -p zoo
   cat > zoo/BUILD <<EOF
java_binary(
    name = "ball-pit",
    srcs = ["BallPit.java"],
    main_class = "BallPit",
    deps = ["//external:mongoose"],
)
EOF

  cat > zoo/BallPit.java <<EOF
import carnivore.Mongoose;

public class BallPit {
    public static void main(String args[]) {
        Mongoose.frolic();
    }
}
EOF

  cat > $build_file <<EOF
java_library(
    name = "mongoose",
    srcs = ["carnivore/Mongoose.java"],
    visibility = ["//visibility:public"],
)
EOF
  bazel run //zoo:ball-pit >& $TEST_log || fail "Failed to build/run zoo"
  expect_log "Tra-la!"

  cat > $package_dir/Mongoose.java <<EOF
package carnivore;
public class Mongoose {
    public static void frolic() {
        System.out.println("Growl!");
    }
}
EOF

  # Check that rebuilding this doesn't rebuild libmongoose.jar, even though it
  # has changed. Bazel assumes that files in external repositories are
  # immutable.
  bazel run //zoo:ball-pit >& $TEST_log || fail "Failed to build/run zoo"
  expect_log "Tra-la!"
  expect_not_log "Building endangered/libmongoose.jar"
  expect_not_log "Growl!"
}

function test_default_ws() {
  bazel build //external:java >& $TEST_log || fail "Failed to build java"
}

run_suite "local repository tests"
