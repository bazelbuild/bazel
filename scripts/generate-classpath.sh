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

source scripts/get_all_bazel_paths.sh || exit 1

cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<classpath>
EOF

for path in ${JAVA_PATHS}; do
    echo "    <classpathentry kind=\"src\" path=\"$path\"/>"
done

# Find third party paths
for path in ${THIRD_PARTY_JAR_PATHS}; do
    echo "    <classpathentry kind=\"lib\" path=\"$path\"/>"
done

# Find protobuf generation
for path in ${PROTOBUF_PATHS}; do
    echo "    <classpathentry kind=\"lib\" path=\"$(dirname $path)/$(basename $path .proto_output)\" sourcepath=\"$path\"/>"
done

for path_pair in ${GENERATED_PATHS}; do
    path_arr=(${path_pair//:/ })
    jar=${path_arr[0]}
    source_path=${path_arr[1]}
    echo "    <classpathentry kind=\"lib\" path=\"${jar}\" sourcepath=\"${source_path}\"/>"
done

# Write the end of the .classpath file
cat <<EOF
    <classpathentry kind="lib" path="tools/jdk/jdk/lib/tools.jar"/>
    <classpathentry kind="output" path="${IDE_OUTPUT_PATH}"/>
    <classpathentry kind="con" path="org.eclipse.jdt.launching.JRE_CONTAINER/org.eclipse.jdt.internal.debug.ui.launcher.StandardVMType/JavaSE-1.8">
      <accessrules>
        <accessrule kind="accessible" pattern="**"/>
      </accessrules>
    </classpathentry>
</classpath>
EOF
