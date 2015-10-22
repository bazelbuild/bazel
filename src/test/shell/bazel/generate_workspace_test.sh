#!/bin/bash
#
# Copyright 2015 The Bazel Authors. All rights reserved.
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
# Test //external mechanisms
#

# Load test environment
src_dir=$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)
source $src_dir/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }
source $src_dir/remote_helpers.sh \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

export JAVA_RUNFILES=$TEST_SRCDIR

function set_up() {
  # Set up custom repository directory.
  m2=$TEST_TMPDIR/my-m2
  rm -rf $m2
  mkdir -p $m2
  startup_server $m2
}

function tear_down() {
  shutdown_server
  rm -rf $m2
}

function generate_workspace() {
  ${bazel_data}/src/tools/generate_workspace/generate_workspace $@
}

# Takes: groupId, artifactId, and version.
function make_artifact() {
  local groupId=$1
  local artifactId=$2
  local version=$3

  local pkg_dir=$m2/$groupId/$artifactId/$version
  mkdir -p $pkg_dir
  # Make the pom.xml.
  cat > $pkg_dir/$artifactId-$version.pom <<EOF
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>$artifactId</groupId>
  <artifactId>$artifactId</artifactId>
  <version>$version</version>
</project>
EOF

  # Make the jar with one class (we use the groupId for the classname).
  cat > $TEST_TMPDIR/$groupId.java <<EOF
public class $groupId {
  public static void print() {
    System.out.println("$artifactId");
  }
}
EOF
  ${bazel_javabase}/bin/javac $TEST_TMPDIR/$groupId.java
  ${bazel_javabase}/bin/jar cf $pkg_dir/$artifactId-$version.jar $TEST_TMPDIR/$groupId.class
}

function get_workspace_file() {
  cat $TEST_log | tail -n 2 | head -n 1
}

function get_build_file() {
  cat $TEST_log | tail -n 1
}

function test_pom() {
  # Create a maven repo
  make_artifact blorp glorp 1.2.3

  # Create a pom that references the artifacts.
  cat > $TEST_TMPDIR/pom.xml <<EOF
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>my</groupId>
  <artifactId>thing</artifactId>
  <version>1.0</version>
  <repositories>
    <repository>
      <id>my-repo1</id>
      <name>a custom repo</name>
      <url>http://localhost:$fileserver_port/</url>
    </repository>
  </repositories>

  <dependencies>
    <dependency>
      <groupId>blorp</groupId>
      <artifactId>glorp</artifactId>
      <version>1.2.3</version>
    </dependency>
  </dependencies>
</project>
EOF

  generate_workspace --maven_project=$TEST_TMPDIR &> $TEST_log \
    || fail "generating workspace failed"

  cat $(cat $TEST_log | tail -n 2 | head -n 1) > ws
  cat $(cat $TEST_log | tail -n 1) > build

  assert_contains "artifact = \"blorp:glorp:1.2.3\"," ws
  assert_contains "repository = \"http://localhost:$fileserver_port/\"," ws
  assert_contains "\"@blorp_glorp//jar\"," build
}

function test_invalid_pom() {
  # No pom file.
  rm -f $TEST_TMPDIR/pom.xml
  generate_workspace -m $TEST_TMPDIR &> $TEST_log
  expect_log "Non-readable POM $TEST_TMPDIR/pom.xml"

  # Invalid XML.
  cat > $TEST_TMPDIR/pom.xml <<EOF
<project>
EOF
  generate_workspace -m $TEST_TMPDIR &> $TEST_log
  expect_log "expected end tag </project>"
}

function test_profile() {
  cat > $TEST_TMPDIR/pom.xml <<EOF
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>my</groupId>
  <artifactId>thing</artifactId>
  <version>1.0</version>
  <profiles>
    <profile>
      <id>my-profile</id>
      <activation>
        <property>
          <name>makeThing</name>
          <value>thing</value>
        </property>
      </activation>
    </profile>
  </profiles>
</project>
EOF

  generate_workspace --maven_project=$TEST_TMPDIR &> $TEST_log \
    || fail "generating workspace failed"
}

function test_submodules() {
  cat > $TEST_TMPDIR/pom.xml <<EOF
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>xyz</groupId>
    <artifactId>a</artifactId>
    <version>1.0</version>
    <packaging>pom</packaging>
    <modules>
        <module>b1</module>
        <module>b2</module>
    </modules>
</project>
EOF

  # Create submodules, version and group are inherited from parent.
  mkdir -p $TEST_TMPDIR/{b1,b2}
  cat > $TEST_TMPDIR/b1/pom.xml <<EOF
<project>
    <modelVersion>4.0.0</modelVersion>
    <artifactId>b1</artifactId>
    <parent>
        <groupId>xyz</groupId>
        <artifactId>a</artifactId>
        <version>1.0</version>
    </parent>
    <dependencies>
        <dependency>
            <groupId>xyz</groupId>
            <artifactId>b2</artifactId>
            <version>1.0</version>
        </dependency>
    </dependencies>
</project>
EOF

  cat > $TEST_TMPDIR/b2/pom.xml <<EOF
<project>
    <modelVersion>4.0.0</modelVersion>
    <artifactId>b2</artifactId>
    <parent>
        <groupId>xyz</groupId>
        <artifactId>a</artifactId>
        <version>1.0</version>
    </parent>
</project>
EOF

  generate_workspace -m $TEST_TMPDIR/b1 &> $TEST_log || fail "generate failed"
  expect_log "xyz_b2 was defined in $TEST_TMPDIR/b2/pom.xml which isn't a repository URL"
  assert_contains "artifact = \"xyz:b2:1.0\"," $(get_workspace_file)
}

run_suite "maven tests"
