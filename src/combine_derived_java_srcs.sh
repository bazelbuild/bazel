#!/bin/sh

# Copyright 2016 The Bazel Authors. All rights reserved.
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

set -eu

# Combine src jars to a single archive containing all the source files.

OUTPUT="${PWD}/$1"
shift

TMP_DIR=${TMPDIR:-/tmp}
PACKAGE_DIR="$(mktemp -d ${TMP_DIR%%/}/bazel.XXXXXXXX)"
trap "rm -fr \"${PACKAGE_DIR}\"" EXIT
JAVA_SRC_DIR="${PACKAGE_DIR}/derived/src/java"
mkdir -p "${JAVA_SRC_DIR}"

for i in $*
do
    JARFILE="${PWD}/$i"
    (cd "${JAVA_SRC_DIR}" && jar xf "${JARFILE}")
done

find "${PACKAGE_DIR}" -exec touch -t 198001010000.00 '{}' '+'
(cd "${PACKAGE_DIR}" && find . -type f | sort | zip -qDX@ "${OUTPUT}")
