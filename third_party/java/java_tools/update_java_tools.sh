#!/bin/bash
#
# A script to update the checked-in jars corresponding to Java tools used
# by the Java rules in Bazel.
#
# For usage please run
# ~/third_party/java/java_tools/update_java_tools.sh help

declare -A tool_name_to_target=(["JavaBuilder"]="src/java_tools/buildjar:JavaBuilder_deploy.jar" \
["VanillaJavaBuilder"]="src/java_tools/buildjar:VanillaJavaBuilder_deploy.jar" \
["GenClass"]="src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass:GenClass_deploy.jar" \
["Runner"]="src/java_tools/junitrunner/java/com/google/testing/junit/runner:Runner_deploy.jar" \
["ExperimentalRunner"]="src/java_tools/junitrunner/java/com/google/testing/junit/runner:ExperimentalRunner_deploy.jar" \
["JacocoCoverage"]="src/java_tools/junitrunner/java/com/google/testing/coverage:JacocoCoverage_jarjar_deploy.jar" \
["Turbine"]="src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:turbine_deploy.jar" \
["TurbineDirect"]="src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:turbine_direct_binary_deploy.jar" \
["SingleJar"]="src/java_tools/singlejar/java/com/google/devtools/build/singlejar:bazel-singlejar_deploy.jar")

usage="This script updates the checked-in jars corresponding to the tools "\
"used by the Java rules in Bazel.

To update all the tools simultaneously run from your bazel workspace root:
~/third_party/java/java_tools/update_java_tools.sh

To update only one or one subset of the tools run
third_party/java/java_tools/update_java_tools.sh tool_1 tool_2 ... tool_n

where tool_i can have one of the values: ${!tool_name_to_target[*]}.

For example, to update only JavaBuilder run
third_party/java/java_tools/update_java_tools.sh JavaBuilder

To update JavaBuilder, Turbine and SingleJar run
third_party/java/java_tools/update_java_tools.sh JavaBuilder Turbine SingleJar
"

if [[ ! -z "$1" && $1 = "help" ]]; then
  echo "$usage"
  exit
fi

tools_to_update=()

if [[ ! -z "$@" ]]
then
   # Update only the tools specified on the command line.
   tools_to_update=("$@")
else
  # If no tools were specified update all of them.
  tools_to_update=(${!tool_name_to_target[*]})
fi

updated_tools=()
not_updated_tools=()

function update_tool() {
  local bazel_target="${1}"; shift
  bazel build "$bazel_target"

  local binary=$(echo "bazel-bin/$bazel_target" | sed 's@:@/@')

  if [[ ! -f "$binary" ]]; then
    binary=$(echo "bazel-genfiles/$bazel_target" | sed 's@:@/@')
  fi

  local tool_basename=$(basename $binary)
  if [[ -f "$binary" ]]; then
    cp -f "$binary" "third_party/java/java_tools/$tool_basename"
    echo "Updated third_party/java/java_tools/$tool_basename"
    updated_tools+=("third_party/java/java_tools/$tool_basename")
  else
    echo "Could not build $bazel_target"
    not_updated_tools+=("third_party/java/java_tools/$tool_basename")
  fi
}

for tool in "${tools_to_update[@]}"
do
  tool_bazel_target=${tool_name_to_target[$tool]}
  update_tool "$tool_bazel_target"
done

echo "......"

if [[ ${#not_updated_tools[@]} -gt 0 ]]; then
  echo "ERROR: THE FOLLOWING TOOLS WERE NOT UPDATED! Please check the above logs."
  ( IFS=$'\n'; echo "${not_updated_tools[*]}" )
fi
if [[ ${#updated_tools[@]} -gt 0 ]]; then
  bazel_version=$(bazel version | grep "Build label" | cut -d " " -f 3)
  git_head=$(git rev-parse HEAD)
  echo ""
  echo "Please copy/paste the following into third_party/java/java_tools/README.md:"
  echo ""
  echo "The following tools were built with bazel $bazel_version at commit $git_head \
by running:
$ third_party/java/java_tools/update_java_tools.sh $@
"
  ( IFS=$'\n'; echo "${updated_tools[*]}" )
fi
