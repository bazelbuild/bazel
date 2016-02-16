#!/bin/sh
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
# Generates an Eclipse project. If a .project is not present, it will creates
# it and it will overwrite any .classpath file present
#
# Usage: ./setup-eclipse.sh
#

set -eu
TARGET=$(echo //src/{main,java_tools,test/{java,cpp}}/... //third_party/... \
               -//third_party/ijar/test/... -//third_party/java/j2objc/...)
JRE="JavaSE-1.8"
PROJECT_NAME="bazel"
OUTPUT_PATH="bazel-out/ide/classes"
GENERATED_PATH="bazel-out/ide/generated"
EXTRA_JARS="bazel-bazel/external/local_jdk/lib/tools.jar"

cd $(dirname $(dirname "$0"))

# Compile bazel
[ -f "output/bazel" ] || ./compile.sh >&2 || exit $?

# Make the script use actual bazel
function bazel() {
  ./output/bazel "$@"
}

#
# End of part specific to bazel
#
source ./scripts/get_project_paths.sh

mkdir -p ${OUTPUT_PATH} ${GENERATED_PATH}

# Overwrite .classpath and .factorypath.
./scripts/eclipse-generate.sh classpath "$JAVA_PATHS" "$LIB_PATHS $EXTRA_JARS" "$JRE" "$OUTPUT_PATH" >.classpath
if [ -n "$PLUGIN_PATHS" ]; then
    ./scripts/eclipse-generate.sh factorypath "$PROJECT_NAME" "$PLUGIN_PATHS" >.factorypath
    mkdir -p .settings
    # Write apt settings if not present.
    [ -e ".settings/org.eclipse.jdt.apt.core.prefs" ] || \
        ./scripts/eclipse-generate.sh apt_settings "$GENERATED_PATH" > .settings/org.eclipse.jdt.apt.core.prefs
fi
# Write .project if not present.
[ -e ".project" ] || \
    ./scripts/eclipse-generate.sh project "$PROJECT_NAME" > .project

echo
echo '***'
echo '*** Eclipse project generated'
echo '***'
echo
echo 'You can now import the bazel project into Eclipse.'
