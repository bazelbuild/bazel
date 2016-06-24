#!/bin/sh

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

set -eu

# This script is used to create the directory tree embedded into the Bazel
# binary that is used as the default source for the @bazel_tools repository.
# It shuffles around files compiled in other rules, then zips them up.

OUTPUT="${PWD}/$1"
shift

TMP_DIR=${TMPDIR:-/tmp}
PACKAGE_DIR="$(mktemp -d ${TMP_DIR%%/}/bazel.XXXXXXXX)"
mkdir -p "${PACKAGE_DIR}"
trap "rm -fr \"${PACKAGE_DIR}\"" EXIT

for i in $*; do

  # xcrunwrapper should come from src/tools/xcode/.  Exclude an
  # xcrunwrapper in tools/objc to avoid conflict.
  if  [ "$i" = "tools/objc/xcrunwrapper.sh" ]
  then
    continue
  fi

  case "$i" in
    *tools/jdk/BUILD*) OUTPUT_PATH=tools/jdk/BUILD ;;
    *JavaBuilder*_deploy.jar) OUTPUT_PATH=tools/jdk/JavaBuilder_deploy.jar ;;
    *turbine_deploy.jar) OUTPUT_PATH=tools/jdk/turbine_deploy.jar ;;
    *javac.jar) OUTPUT_PATH=third_party/java/jdk/langtools/javac.jar ;;
    *javac7.jar) OUTPUT_PATH=third_party/java/jdk/langtools/javac7.jar ;;
    *SingleJar_deploy.jar) OUTPUT_PATH=tools/jdk/SingleJar_deploy.jar ;;
    *GenClass_deploy.jar) OUTPUT_PATH=tools/jdk/GenClass_deploy.jar ;;
    *Runner_deploy.jar) OUTPUT_PATH=tools/jdk/TestRunner_deploy.jar ;;
    *ijar.exe) OUTPUT_PATH=tools/jdk/ijar/ijar.exe ;;
    *ijar) OUTPUT_PATH=tools/jdk/ijar/ijar ;;
    *zipper.exe) OUTPUT_PATH=tools/zip/zipper/zipper.exe ;;
    *zipper) OUTPUT_PATH=tools/zip/zipper/zipper ;;
    *src/objc_tools/*) OUTPUT_PATH=tools/objc/precomp_${i##*/} ;;
    *xcode*StdRedirect.dylib) OUTPUT_PATH=tools/objc/StdRedirect.dylib ;;
    *xcode*realpath) OUTPUT_PATH=tools/objc/realpath ;;
    *xcode*xcode-locator) OUTPUT_PATH=tools/objc/xcode-locator ;;
    *src/tools/xcode/*) OUTPUT_PATH=tools/objc/${i##*/}.sh ;;
    *) OUTPUT_PATH=$(echo $i | sed 's_^.*bazel-out/[^/]*/bin/__') ;;
  esac

  mkdir -p "${PACKAGE_DIR}/$(dirname "${OUTPUT_PATH}")"
  cp "$i" "${PACKAGE_DIR}/${OUTPUT_PATH}"
done

if [ ! -f ${PACKAGE_DIR}/third_party/java/jdk/langtools/javac.jar ]; then
  cp ${PACKAGE_DIR}/third_party/java/jdk/langtools/javac7.jar \
      ${PACKAGE_DIR}/third_party/java/jdk/langtools/javac.jar
fi

cat > "${PACKAGE_DIR}/WORKSPACE" <<EOF
workspace(name = "bazel_tools")
EOF
mkdir -p "${PACKAGE_DIR}/tools/defaults"
touch "${PACKAGE_DIR}/tools/defaults/BUILD"
for i in $(find "${PACKAGE_DIR}" -name BUILD.tools); do
  mv "$i" "$(dirname "$i")/BUILD"
done
find "${PACKAGE_DIR}" -exec touch -t 198001010000.00 '{}' '+'
(cd "${PACKAGE_DIR}" && find . -type f | sort | zip -qDX@ "${OUTPUT}")
