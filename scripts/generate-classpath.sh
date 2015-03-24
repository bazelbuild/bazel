#!/bin/bash
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
# Generates an eclipse .classpath file for Bazel

set -eu

cd $(dirname "$0")
cd ..

function query() {
    ./output/bazel query "$@"
}

# Compile bazel
([ -f "output/bazel" ] && [ -f "tools/jdk/JavaBuilder_deploy.jar" ] && [ -f "tools/jdk/ijar" ] \
    && [ -f "tools/jdk/SingleJar_deploy.jar" ] && [ -e "tools/jdk/jdk" ]) || ./compile.sh >&2 || exit $?

# Build everything
./output/bazel build //src/... //third_party/... >&2 || exit $?

cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<classpath>
EOF

# Find Java paths
JAVA_PATHS="$(find src -name "*.java" | sed "s|/com/google/.*$||" | sort -u)"
# TODO(bazel-team): Once objc_tools have buildfiles, uncomment the if below
# if [ "$(uname -s | tr 'A-Z' 'a-z')" != "darwin" ]; then
JAVA_PATHS="$(echo "${JAVA_PATHS}" | fgrep -v "/objc_tools/")"
# fi
for path in ${JAVA_PATHS}; do
    echo "    <classpathentry kind=\"src\" path=\"$path\"/>"
done

# Find third party paths
for path in $(find third_party -name "*.jar" | sort -u); do
    echo "    <classpathentry kind=\"lib\" path=\"$path\"/>"
done

# Find protobuf generation
for path in $(find bazel-bin/ -name "*.java" | grep proto | sed "s|/com/google/.*$||" | sort -u | sed 's|//|/|'); do
    echo "    <classpathentry kind=\"lib\" path=\"$(dirname $path)/$(basename $path .proto_output)\" sourcepath=\"$path\"/>"
done

# Find other generation
PACKAGE_LIST=$(find src -name "BUILD" | sed "s|/BUILD||" | sed "s|^|//|")
# Returns the package of file $1
function get_package_of() {
  # look for the longest matching package
  for i in ${PACKAGE_LIST}; do
    if [[ "$1" =~ ^$i ]]; then  # we got a match
      echo $(echo -n $i | wc -c | xargs echo) $i
    fi
  done | sort -r -n | head -1 | cut -d " " -f 2
}

# returns the target corresponding to file $1
function get_target_of() {
    local package=$(get_package_of $1)
    local file=$(echo $1 | sed "s|^${package}/||g")
    echo "${package}:${file}"
}

# Returns the target that consume file $1
function get_consuming_target() {
    # Here to the god of bazel, I should probably offer one or two memory chips for that
    local target=$(get_target_of $1)
    local generating_target=$(query "deps(${target}, 1) - ${target}")
    local java_library=$(query "rdeps(//src/..., ${generating_target}, 1) - ${generating_target}")
    echo "${java_library}"
}

# Returns the library that contains the generated file $1
function get_containing_library() {
    get_consuming_target $1 | sed 's|:|/lib|' | sed 's|^//|bazel-bin/|' | sed 's|$|.jar|'
}

for path in $(find bazel-genfiles/ -name "*.java" | sed 's|/\{0,1\}bazel-genfiles/\{1,2\}|//|'); do
    source_path=$(echo $path | sed 's|//|bazel-genfiles/|' | sed 's|/com/.*$||')
    echo "    <classpathentry kind=\"lib\" path=\"$(get_containing_library ${path})\" sourcepath=\"${source_path}\"/>"
done | sort -u

# Write the end of the .classpath file
cat <<'EOF'
    <classpathentry kind="lib" path="tools/jdk/jdk/lib/tools.jar"/>
    <classpathentry kind="output" path="bazel-out/eclipse-classes"/>
    <classpathentry kind="con" path="org.eclipse.jdt.launching.JRE_CONTAINER/org.eclipse.jdt.internal.debug.ui.launcher.StandardVMType/JavaSE-1.8">
      <accessrules>
        <accessrule kind="accessible" pattern="**"/>
      </accessrules>
    </classpathentry>
</classpath>
EOF
