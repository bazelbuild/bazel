#!/bin/bash

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

# Set-up the base workspace, currently used as package_path to provide
# the tools directory.

# Create symlinks so we can use tools and examples from the base_workspace.
function symlink_directory() {
  local dir=$1
  rm -fr "${base_workspace}/${dir}"
  mkdir "${base_workspace}/${dir}"

  for i in $(cd ${dir}; ls | grep -v "^BUILD$"); do
    ln -s "$(pwd)/${dir}/$i" "${base_workspace}/${dir}/$i";
  done

  touch "${base_workspace}/${dir}/DONT_FOLLOW_SYMLINKS_WHEN_TRAVERSING_THIS_DIRECTORY_VIA_A_RECURSIVE_TARGET_PATTERN"
}

base_workspace="${WORKSPACE_DIR}/base_workspace"
mkdir -p "${base_workspace}"

symlink_directory tools
cat > "${base_workspace}/tools/BUILD" <<EOF
package(default_visibility = ["//visibility:public"])
exports_files(["test_sharding_compliant"])
EOF

symlink_directory examples
symlink_directory third_party

# Create a bazelrc file with the base_workspace directory in the package path.
bazelrc='build --package_path %workspace%:'${base_workspace}
bazelrc="${bazelrc}"$'\nfetch --package_path %workspace%:'${base_workspace}
bazelrc="${bazelrc}"$'\nquery --package_path %workspace%:'${base_workspace}
if [ -z "${HOME-}" ]; then
  warning="$INFO No \$HOME variable set, cannot write .bazelrc file."
  warning="$warning Consider adding $base_workspace to your package path"
  display $warning
elif [ ! -f $HOME/.bazelrc ]; then
  display "$INFO Creating a .bazelrc pointing to $base_workspace"
  echo -e "$bazelrc" > $HOME/.bazelrc
else
  while read rcline; do
    if ! grep -q "$rcline" $HOME/.bazelrc; then
      warning="$INFO You already have a .bazelrc. Make sure it contains the "
      warning="$warning following package paths:\n\n$bazelrc\n\n"
      display "$warning"
      break
    fi
  done <<< "$bazelrc"
fi
