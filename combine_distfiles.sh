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

# Combine the archives passed to a single archive.
# NOTE: This assumes that the individual archives are already packed
# in a way to contain canonical timestamps. This assumption must be
# met in order to obtain reproducible output; the assumption is met
# for the source tree and the archive of the generated java files.

OUTPUT="${PWD}/$1"
shift

TMP_DIR=${TMPDIR:-/tmp}
PACKAGE_DIR="$(mktemp -d ${TMP_DIR%%/}/bazel.XXXXXXXX)"
trap "rm -fr \"${PACKAGE_DIR}\"" EXIT
mkdir -p "${PACKAGE_DIR}"

for i in $*
do
    ARCHIVE="${PWD}/$i"
    case "$i" in
        *.zip) UNPACK="unzip -q" ;;
        *.tar) UNPACK="tar xf" ;;
    esac
    (cd "${PACKAGE_DIR}" && ${UNPACK} "${ARCHIVE}")
done

(cd "${PACKAGE_DIR}" && find . -type f | sort | zip -qDX@ "${OUTPUT}")
