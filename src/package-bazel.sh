#!/bin/bash
#
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

set -euo pipefail

# This script creates the Bazel archive that Bazel client unpacks and then
# starts the server from.

WORKDIR=$(pwd)
OUT=$1
EMBEDDED_TOOLS=$2
DEPLOY_JAR=$3
INSTALL_BASE_KEY=$4
PLATFORMS_ARCHIVE=$5
shift 4

if [[ "$OUT" == *jdk_allmodules.zip ]]; then
  DEV_BUILD=1
else
  DEV_BUILD=0
fi

TMP_DIR=${TMPDIR:-/tmp}
PACKAGE_DIR="$(mktemp -d ${TMP_DIR%%/}/bazel.XXXXXXXX)"
mkdir -p "${PACKAGE_DIR}"
trap "rm -fr ${PACKAGE_DIR}" EXIT

cp $* ${PACKAGE_DIR}

if [[ $DEV_BUILD -eq 0 ]]; then
  # Unpack the deploy jar for postprocessing and for "re-compressing" to save
  # ~10% of final binary size.
  unzip -q -d recompress ${DEPLOY_JAR}
  cd recompress

  # Zero out timestamps and sort the entries to ensure determinism.
  find . -type f -print0 | xargs -0 touch -t 198001010000.00
  find . -type f | sort | zip -q0DX@ ../deploy-uncompressed.jar

  # While we're in the deploy jar, grab the label and pack it into the final
  # packaged distribution zip where it can be used to quickly determine version
  # info.
  bazel_label="$(\
    (grep '^build.label=' build-data.properties | cut -d'=' -f2- | tr -d '\n') \
        || echo -n 'no_version')"
  echo -n "${bazel_label:-no_version}" > "${PACKAGE_DIR}/build-label.txt"

  cd ..

  DEPLOY_JAR="deploy-uncompressed.jar"
fi

# The server jar needs to be the first binary we extract. This is how the Bazel
# client knows what .jar to pass to the JVM.
cp ${DEPLOY_JAR} ${PACKAGE_DIR}/A-server.jar
cp ${INSTALL_BASE_KEY} ${PACKAGE_DIR}/install_base_key
# The timestamp of embedded tools should already be zeroed out in the input zip
touch -t 198001010000.00 ${PACKAGE_DIR}/*

if [ -n "${EMBEDDED_TOOLS}" ]; then
  mkdir ${PACKAGE_DIR}/embedded_tools
  (cd ${PACKAGE_DIR}/embedded_tools && unzip -q "${WORKDIR}/${EMBEDDED_TOOLS}")
fi

# Unzip platforms.zip into platforms/, move files up from external/platforms
# subdirectory, and cleanup after itself.
( \
  cd ${PACKAGE_DIR} && \
    unzip -q -d platforms platforms.zip && \
    rm platforms.zip && \
    cd platforms && \
    mv external/platforms/* . && \
    rmdir -p external/platforms \
)
touch -t 198001010000.00 ${PACKAGE_DIR}/platforms/WORKSPACE

(cd ${PACKAGE_DIR} && find . -type f | sort | zip -q9DX@ "${WORKDIR}/${OUT}")
