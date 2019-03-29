#!/bin/bash

set -euo pipefail

VERSION="0.1"
VERSIONED_FILENAME="android_tools_pkg-$VERSION.tar.gz"

android_tools_archive="tools/android/runtime_deps/android_tools.tar.gz"
versioned_android_tools_archive="/tmp/$VERSIONED_FILENAME"

rm -f $versioned_android_tools_archive

cp $android_tools_archive $versioned_android_tools_archive
tar tf $versioned_android_tools_archive

gsutil cp -a public-read $versioned_android_tools_archive gs://bazel-mirror/bazel_android_tools/$VERSIONED_FILENAME
