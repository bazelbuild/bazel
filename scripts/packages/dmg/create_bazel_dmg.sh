#!/usr/bin/env bash

# Copyright 2020 The Bazel Authors. All rights reserved.
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

# The script for creating a dmg file for Bazel installation on macOS.

set -e

for i in "$@"
do
case $i in
    # The Bazel binary we are going to ship
    --bazel_binary=*)
    BAZEL_BINARY="${i#*=}"
    ;;

    # (Optional) The macOS resource file for Bazel icon (.rsrc)
    --bazel_icon*)
    BAZEL_ICON="${i#*=}"
    ;;

    # (Optional) The .DS_STORE we use for the dmg file, which contains
    # information about the layout and background of the dmg disk.
    --ds_store=*)
    DS_STORE="${i#*=}"
    ;;

    # (Optional) The background we use when opening the dmg file.
    # It will only work with a corresponding .DS_STORE file.
    --background=*)
    BACKGROUND="${i#*=}"
    ;;

    # The output dmg file
    --output=*)
    OUTPUT="${i#*=}"
    ;;

    *)
    echo "Warning: Unknown option ${i}"
    ;;
esac
done

if [ -z "${BAZEL_BINARY}" ]; then
    echo "Error: Bazel binary must be specified by --bazel_binary"
    exit 1
fi

if [ -z "${OUTPUT}" ]; then
    echo "Error: Output dmg file must be specified by --output"
    exit 1
fi

TMP_DIR="$(mktemp -d -t bazel-dmg-XXXX)"
trap "rm -rf ${TMP_DIR}" EXIT

cp "${BAZEL_BINARY}" "${TMP_DIR}/bazel"
ln -s /usr/local/bin "${TMP_DIR}/bin"

# Set Bazel icon
if [ -n "${BAZEL_ICON}" ]; then
    chmod +w "${TMP_DIR}/bazel"
    # Append icon resource file for the Bazel binary
    Rez -append "${BAZEL_ICON}" -o "${TMP_DIR}/bazel"
    # Set Custom icon attribute for the Bazel binary
    SetFile -a C "${TMP_DIR}/bazel"
    chmod 0755 "${TMP_DIR}/bazel"
fi

# Use a preconfigured .DS_STORE and set the background image
if [ -n "${DS_STORE}" ]; then
    cp "${DS_STORE}" "${TMP_DIR}/.DS_Store"

    if [ -n "${BACKGROUND}" ]; then
        mkdir "${TMP_DIR}/.background"
        cp "${BACKGROUND}" "${TMP_DIR}/.background/"
    fi
fi

# Create the dmg file
hdiutil create "${OUTPUT}" -ov -volname "bazel-install" -format UDZO -fs HFS+ -srcfolder "${TMP_DIR}"

