#!/bin/bash

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

update_tool "src/java_tools/buildjar:JavaBuilder_deploy.jar"
update_tool "src/java_tools/buildjar:VanillaJavaBuilder_deploy.jar"
update_tool "src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass:GenClass_deploy.jar"
update_tool "src/java_tools/junitrunner/java/com/google/testing/junit/runner:Runner_deploy.jar"
update_tool "src/java_tools/junitrunner/java/com/google/testing/junit/runner:ExperimentalRunner_deploy.jar"
update_tool "src/java_tools/junitrunner/java/com/google/testing/coverage:JacocoCoverage_jarjar_deploy.jar"

bazel_version=$(bazel version | grep "Build label" | cut -d " " -f 3)
git_head=$(git rev-parse HEAD)
echo "......"
echo ""
echo "The following tools were built with bazel $bazel_version at commit $git_head"
( IFS=$'\n'; echo "${tools[*]}" )
