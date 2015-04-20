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
# Gets all libraries needed for IDE support of a Bazel workspace

set -eu

cd $(dirname "$0")
cd ..

function query() {
    ./output/bazel query "$@"
}

# Compile bazel
([ -f "output/bazel" ] && [ -f "tools/jdk/JavaBuilder_deploy.jar" ] \
    && [ -f "tools/jdk/ijar" ] && [ -f "tools/jdk/SingleJar_deploy.jar" ] \
    && [ -e "tools/jdk/jdk" ]) || ./compile.sh >&2 || exit $?

# Build everything
./output/bazel build ${TARGET} >&2 || exit $?

# Now for java each targets, find all sources and all jars
DEPS=$(query 'filter("\.java$",
                    deps(kind("(java_binary|java_library|java_test|java_plugin)",
                         deps('"$TARGET"')))
                    except deps(//tools/...))')
PATHS=$(echo "$DEPS" | sed 's|:|/|' | sed 's|^//||')

# Java Files:
JAVA_PATHS=$(echo "$PATHS" | sed 's_\(/java\(tests\)\{0,1\}\)/.*$_\1_' | sort -u)

# Java plugins
JAVA_PLUGINS_DEPS=$(query 'filter("\.jar$",
                                  deps(kind(java_import,
                                            deps(kind(java_plugin,
                                                     deps('"$TARGET"')))))
                                  except deps(//tools/...))')
PLUGIN_PATHS=$(echo "$JAVA_PLUGINS_DEPS" | sed 's|:|/|' | sed 's|^//||' | sort -u)

# Jar Files:
JAR_DEPS=$(query 'filter("\.jar$", deps(kind(java_import, deps('"$TARGET"')))
                                   except deps(//tools/...))')
JAR_FILES=$(echo "$JAR_DEPS" | sed 's|:|/|' | sed 's|^//||' | sort -u)

# Generated files are direct dependencies of java rules that are not java rules,
# filegroup or binaries.
# We also handle genproto separately it is output in bazel-genfiles not in
# bazel-bin.
# We suppose that all files are generated in the same package than the library.
GEN_LIBS=$(query 'let gendeps = kind(rule, deps(kind(java_*, deps('"$TARGET"')), 1))
                              - kind("(java_.*|filegroup|.*_binary|genproto)", deps('"$TARGET"'))
                              - deps(//tools/...)
                  in rdeps('"$TARGET"', set($gendeps), 1) - set($gendeps)' \
    | sed 's|^//\(.*\):\(.*\)|bazel-bin/\1/lib\2.jar:bazel-genfiles/\1|')

# Hack for genproto
PROTOBUFS=$(bazel query 'kind(genproto, deps('"$TARGET"'))' \
    | sed 's|^//\(.*\):\(.*\)$|bazel-bin/\1/lib\2.jar:bazel-bin/\1/lib\2.jar.proto_output/|')
LIB_PATHS="${JAR_FILES} ${PROTOBUFS} ${GEN_LIBS}"
