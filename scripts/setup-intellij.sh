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
# Generates an IntelliJ project in Bazel.

set -o errexit
cd $(dirname "$0")
cd ..

mkdir -p .idea/
cp -R scripts/resources/idea/*.* .idea/
source scripts/get_all_bazel_paths.sh


readonly iml_file=bazel.iml
# The content root output/classes is used for generated sources, specifically
# AutoValue.
cat > $iml_file <<EOH
<?xml version="1.0" encoding="UTF-8"?>
<module type="JAVA_MODULE" version="4">
  <component name="NewModuleRootManager">
    <output url="file://\$MODULE_DIR\$/${IDE_OUTPUT_PATH}" />
    <exclude-output />
    <orderEntry type="inheritedJdk" />
    <content url="file://\$MODULE_DIR\$/output/classes">
      <sourceFolder url="file://\$MODULE_DIR\$/output/classes" isTestSource="false" />
      <excludeFolder url="file://\$MODULE_DIR\$/output/classes/org" />
    </content>
    <content url="file://\$MODULE_DIR\$/src">
EOH

for source in ${JAVA_PATHS}; do
     if [[ $source == *"javatests" ]]; then
       is_test_source="true"
     elif [[ $source == *"test/java" ]]; then
       is_test_source="true"
     else
       is_test_source="false"
     fi
     echo '      <sourceFolder url="file://$MODULE_DIR$/'"${source}\" isTestSource=\"${is_test_source}\" />" >> $iml_file
done
cat >> $iml_file <<'EOF'
    </content>
    <content url="file://$MODULE_DIR$/third_party/java">
EOF

THIRD_PARTY_JAVA_PATHS="$(ls third_party/java | sort -u | sed -e 's%$%/java%')"

for third_party_java_path in ${THIRD_PARTY_JAVA_PATHS}; do
  echo '      <sourceFolder url="file://$MODULE_DIR$/third_party/java/'${third_party_java_path}'" isTestSource="false" />' >> $iml_file
done
cat >> $iml_file <<'EOF'
    </content>
    <orderEntry type="sourceFolder" forTests="false" />
EOF

function write_jar_entry() {
  local jar_file=$1
  if [[ $# > 1 ]]; then
    local source_path=$2
  else
    local source_path=""
  fi
  local readonly basename=${jar_file##*/}
    cat >> $iml_file <<EOF
      <orderEntry type="module-library">
        <library name="${basename}">
          <CLASSES>
            <root url="jar://\$MODULE_DIR\$/${jar_file}!/" />
          </CLASSES>
          <JAVADOC />
EOF
  if [[ -z "${source_path}" ]]; then
    echo "          <SOURCES />" >> $iml_file
  else
    cat >> $iml_file <<EOF
          <SOURCES>
            <root url="jar:/\$MODULE_DIR\$/${source_path}!/" />
          </SOURCES>
EOF
  fi
  cat >> $iml_file <<'EOF'
      </library>
    </orderEntry>
EOF
}

for jar in ${THIRD_PARTY_JAR_PATHS}; do
  write_jar_entry $jar
done

for source_path in ${PROTOBUF_PATHS}; do
  write_jar_entry ${source_path%.proto_output} $source_path
done

for path_pair in ${GENERATED_PATHS}; do
  write_jar_entry ${path_pair//:/ }
done

write_jar_entry tools/jdk/jdk/lib/tools.jar
cat >> $iml_file <<'EOF'
  </component>
</module>
EOF
