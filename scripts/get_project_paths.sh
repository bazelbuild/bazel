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
#
# Gets all libraries needed for IDE support of a Bazel workspace

set -eu

# Build everything
bazel build -- ${TARGET} >&2 || exit $?

function query() {
    bazel query -k -- "$@"
}

# Find the bazel-workspaceName link
EXECUTION_ROOT_PATH=$(bazel info execution_root)
WORKSPACE_PATH=$(bazel info workspace)
for i in ${WORKSPACE_PATH}/bazel-*; do
  if [[ "$(readlink $i)" == "${EXECUTION_ROOT_PATH}" ]]; then
    EXECUTION_ROOT=$(basename $i)
  fi
done

# Do a bazel query and replace the result by paths relative to the workspace.
#   @repo//package:target will be replaced by
#                         bazel-%workspaceName%/external/repo/package/target
#   //package:target will be replaced by package/target
function query_to_path() {
   query "$1" | sed 's|:|/|' \
     | sed 's|@\(.*\)///\{0,1\}|'"${EXECUTION_ROOT}"'/external/\1/|' \
     | sed 's|^//||' | sort -u
}

ACTUAL_TARGETS="set($(query $(echo ${TARGET} | sed 's/ \([^-]\)/ +\1/g')))"

# Now for java each targets, find all sources and all jars
PATHS=$(query_to_path 'filter("\.java$",
                    deps(kind("(java_binary|java_library|java_test|java_plugin)",
                         deps('"$ACTUAL_TARGETS"')))
                    except deps(//tools/...))')
# Java Files:
JAVA_PATHS=$(echo "$PATHS" | sed 's_\(/java\(tests\)\{0,1\}\)/.*$_\1_' | sort -u)

# Java plugins
PLUGIN_PATHS=$(query_to_path 'filter("\.jar$",
                                     deps(kind(java_import,
                                               deps(kind(java_plugin,
                                                         deps('"$ACTUAL_TARGETS"')))))
                                     except deps(//tools/...))')
# Jar Files:
JAR_FILES=$(query_to_path 'filter("\.jar$", deps(kind(java_import, deps('"$ACTUAL_TARGETS"')))
                                            except deps(//tools/...))')

# Generated files are direct dependencies of java rules that are not java rules,
# filegroup, binaries or external dependencies.
# We also handle genproto separately it is output in bazel-genfiles not in
# bazel-bin.
# We suppose that all files are generated in the same package than the library.
GEN_LIBS=$(query 'let gendeps = kind(rule, deps(kind(java_*, deps('"$ACTUAL_TARGETS"')), 1))
                              - kind("(java_.*|filegroup|.*_binary|genproto|bind)", deps('"$ACTUAL_TARGETS"'))
                              - deps(//tools/...)
                  in rdeps('"$ACTUAL_TARGETS"', set($gendeps), 1) - set($gendeps)' \
    | sed 's|^//\(.*\):\(.*\)|bazel-bin/\1/lib\2.jar:bazel-genfiles/\1|')

# Hack for genproto
PROTOBUFS=$(query 'kind(genproto, deps('"$ACTUAL_TARGETS"'))' \
    | sed 's|^//\(.*\):\(.*\)$|bazel-bin/\1/lib\2.jar:bazel-bin/\1/lib\2.jar.proto_output/|')
LIB_PATHS="${JAR_FILES} ${PROTOBUFS} ${GEN_LIBS}"
