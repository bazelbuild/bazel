#!/bin/bash
#
# A script to update the checked-in jars corresponding to Java tools used
# by the Java rules in Bazel.
#
# For usage please run
# ~/third_party/java/java_tools/update_java_tools.sh help

declare -a all_tools=("JavaBuilder" "VanillaJavaBuilder" "GenClass" "Runner" "ExperimentalRunner" "JacocoCoverage" "Turbine" "TurbineDirect" "SingleJar")

declare -A tool_name_to_target=( ["JavaBuilder"]="src/java_tools/buildjar:JavaBuilder_deploy.jar" \
["VanillaJavaBuilder"]="src/java_tools/buildjar:VanillaJavaBuilder_deploy.jar" \
["GenClass"]="src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass:GenClass_deploy.jar" \
["Runner"]="src/java_tools/junitrunner/java/com/google/testing/junit/runner:Runner_deploy.jar" \
["ExperimentalRunner"]="src/java_tools/junitrunner/java/com/google/testing/junit/runner:ExperimentalRunner_deploy.jar" \
["JacocoCoverage"]="src/java_tools/junitrunner/java/com/google/testing/coverage:JacocoCoverage_jarjar_deploy.jar" \
["Turbine"]="src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:turbine_deploy.jar" \
["TurbineDirect"]="src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:turbine_direct_binary_deploy.jar" \
["SingleJar"]="src/java_tools/singlejar:SingleJar_deploy.jar")

usage="This script updates the checked-in jars corresponding to the tools "\
"used by the Java rules in Bazel.

To update all the tools simultaneously run from your bazel workspace root:
~/third_party/java/java_tools/update_java_tools.sh

To update only one or one subset of the tools run
~/third_party/java/java_tools/update_java_tools.sh tool_1 tool_2 ... tool_n

where tool_i can have one of the values ${all_tools[@]}

For example, to update only JavaBuilder run
~/third_party/java/java_tools/update_java_tools.sh JavaBuilder

To update JavaBuilder, Turbine and SingleJar run
~/third_party/java/java_tools/update_java_tools.sh JavaBuilder Turbine SingleJar
"

if [[ ! -z "$1" && $1 = "help" ]]; then
  echo "$usage"
  exit
fi

tools_to_update=()

if [[ ! -z "$@" ]]
then
   tools_to_update=("$@")
else
#  tools_to_update=("${JAVA_TOOLS_TARGETS[@]}")
  tools_to_update=("${all_tools[@]}")
fi


tools=()

function update_tool() {
  local bazel_target="${1}"; shift
  local binary=$(echo "bazel-bin/$bazel_target" | sed 's@:@/@')

  bazel build "$bazel_target"

  local tool_basename=$(basename $binary)
  cp -f "$binary" "third_party/java/java_tools/$tool_basename"
  echo "Updated third_party/java/java_tools/$tool_basename"
  tools+=("third_party/java/java_tools/$tool_basename")
}

for tool in "${tools_to_update[@]}"
do
  tool_bazel_target=${tool_name_to_target[$tool]}
  update_tool "$tool_bazel_target"
done

bazel_version=$(bazel version | grep "Build label" | cut -d " " -f 3)
git_head=$(git rev-parse HEAD)
echo "......"
echo ""

echo "The following tools were built with bazel $bazel_version at commit $git_head"
( IFS=$'\n'; echo "${tools[*]}" )
