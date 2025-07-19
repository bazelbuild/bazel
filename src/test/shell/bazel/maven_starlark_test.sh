#!/usr/bin/env bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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
# Test the Starlark implementation of the maven_jar() rule.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source ${CURRENT_DIR}/remote_helpers.sh \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

function setup_zoo() {
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
}

# This function takes an optional url argument: mirror. If one is passed, a
# mirror of maven central will be set for the url. It also creates a `BUILD`
# file in the same directory as $local_maven_settings so that it can be used as
# a label in WORKSPACE.
function setup_local_maven_settings_xml() {
  local_maven_settings_xml=settings.xml
  touch $(pwd)/BUILD
  cat > $local_maven_settings_xml <<EOF
<!-- # DO NOT EDIT: automatically generated settings.xml for maven_dependency_plugin -->
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0
    https://maven.apache.org/xsd/settings-1.0.0.xsd">
  <localRepository>$TEST_SRCDIR/m2/repository</localRepository>
EOF
  if [ "$#" -eq 1 ]; then
    cat >> $local_maven_settings_xml <<EOF
  <mirrors>
    <mirror>
      <id>central</id>
      <url>$1</url>
      <mirrorOf>*,default</mirrorOf>
    </mirror>
  </mirrors>
EOF
  fi
  cat >> $local_maven_settings_xml <<EOF
</settings>
EOF
}

function tear_down() {
  shutdown_server
}

function test_maven_jar_starlark() {
  setup_zoo
  version="1.21"
  serve_artifact com.example.carnivore carnivore $version
  setup_local_maven_settings_xml "http://localhost:$fileserver_port"

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:maven_rules.bzl", "maven_jar")
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:$version",
    sha1 = '$sha1',
    settings = '//:$local_maven_settings_xml',
)

bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  expect_log "Tra-la!"
}

function DISABLEDtest_maven_jar_with_classifier_starlark() {
  setup_zoo
  version="1.21"
  packaging="jar"
  classifier="sources"
  serve_artifact com.example.carnivore carnivore $version $packaging $classifier
  setup_local_maven_settings_xml "http://localhost:$fileserver_port"

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:maven_rules.bzl", "maven_jar")
maven_jar(
    name = 'bar_sources',
    artifact = "com.example.foo:bar:$version:jar:sources",
    sha1 = '$sha1',
    settings = '//:$local_maven_settings_xml',
)

bind(name = 'baz_sources', actual = '@bar_sources//jar')
EOF

  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  expect_log "Tra-la!"
}

function setup_android_binary() {
  mkdir -p java/com/app
  cat > java/com/app/BUILD <<EOF
android_binary(
    name = "app",
    manifest = "AndroidManifest.xml",
    deps = ["@herbivore//aar"],
)
EOF
  cat > java/com/app/AndroidManifest.xml <<EOF
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.app"
    android:versionCode="1"
    android:versionName="1.0" >
    <application />
    <uses-sdk
        android:minSdkVersion="9"
        android:targetSdkVersion="20" />
</manifest>
EOF
}

function test_maven_aar_starlark() {
  setup_android_sdk_support
  if [[ ! -d "${TEST_SRCDIR}/androidsdk" ]]; then
    fail "This test cannot run without android_sdk_repository set up," \
      "see the WORKSPACE file for instructions"
  fi
  setup_android_binary
  serve_artifact com.example.carnivore herbivore 1.21 aar
  setup_local_maven_settings_xml "http://localhost:$fileserver_port"
  cat >> WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:maven_rules.bzl", "maven_aar")
maven_aar(
    name = "herbivore",
    artifact = "com.example.carnivore:herbivore:1.21",
    sha1 = "$sha1",
    settings = "//:$local_maven_settings_xml",
    deps = ["@herbivore2//aar"],
)
maven_aar(
    name = "herbivore2",
    artifact = "com.example.carnivore:herbivore:1.21",
    sha1 = "$sha1",
    settings = "//:$local_maven_settings_xml",
)
EOF
  bazel build //java/com/app || fail "Expected build to succeed"
  unzip -l bazel-bin/java/com/app/app.apk > $TEST_log
  expect_log_once "res/layout/my_view.xml"
  unzip -l bazel-bin/java/com/app/app_deploy.jar > $TEST_log
  expect_log_once "com/herbivore/Stegosaurus.class"
  bazel query 'deps(//java/com/app)' >& $TEST_log
  expect_log "@herbivore//aar:aar"
  expect_log "@herbivore2//aar:aar"
}

# Same as test_maven_jar, except omit sha1 implying "we don't care".
function test_maven_jar_no_sha1_starlark() {
  setup_zoo
  version="1.22"
  serve_artifact com.example.carnivore carnivore $version
  setup_local_maven_settings_xml "http://localhost:$fileserver_port/"

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:maven_rules.bzl", "maven_jar")

maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:$version",
    settings = '//:$local_maven_settings_xml',
)

bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  expect_log "Tra-la!"
}

function test_maven_jar_404_starlark() {
  setup_zoo
  version="1.23"
  serve_not_found
  setup_local_maven_settings_xml "http://localhost:$nc_port/",

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:maven_rules.bzl", "maven_jar")
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:$version",
    settings = '//:$local_maven_settings_xml',
)

bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel clean --expunge
  bazel build //zoo:ball-pit >& $TEST_log && echo "Expected build to fail"
  kill_nc
  expect_log "Failed to fetch Maven dependency"
}

function test_maven_jar_mismatched_sha1_starlark() {
  setup_zoo
  version="1.24"
  serve_artifact com.example.carnivore carnivore 1.24
  setup_local_maven_settings_xml "http://localhost:$fileserver_port/"

  wrong_sha1="0123456789012345678901234567890123456789"
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:maven_rules.bzl", "maven_jar")

maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.24",
    sha1 = '$wrong_sha1',
    settings = '//:$local_maven_settings_xml',
)

bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel fetch //zoo:ball-pit >& $TEST_log && echo "Expected fetch to fail"
  expect_log "has SHA-1 of $sha1, does not match expected SHA-1 ($wrong_sha1)"
}

function test_unimplemented_server_attr_starlark() {
  setup_zoo
  version="1.25"
  serve_jar
  setup_local_maven_settings_xml

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:maven_rules.bzl", "maven_jar")

maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:$version",
    server = "attr_not_implemented",
    settings = "//:$local_maven_settings_xml",
)

bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel build //zoo:ball-pit >& $TEST_log && echo "Expected build to fail"
  kill_nc
  expect_log "specifies a 'server' attribute which is currently not supported."
}

run_suite "maven Starlark tests"
