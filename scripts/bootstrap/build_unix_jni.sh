#!/usr/bin/env bash

# Copyright 2026 The Bazel Authors. All rights reserved.
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

JAVA_HOME="$1"
PLATFORM="$2"
OUT="$3"

# Source files common to all platforms.
# Omit blake3, which would require an external dependency.
SOURCES=(
  src/main/native/common.cc
  src/main/native/latin1_jni_path.cc
  src/main/native/process.cc
  src/main/native/unix_jni.cc
  src/main/cpp/util/logging.cc
)

# Compiler flags common to all platforms.
FLAGS=("-std=c++17" "-I." "-I${JAVA_HOME}/include" "-fPIC" "-shared")

# Platform-specific source files and compiler flags.
case "$PLATFORM" in
linux)
  SOURCES+=(src/main/native/unix_jni_linux.cc)
  FLAGS+=("-I${JAVA_HOME}/include/linux")
  ;;
darwin)
  SOURCES+=(src/main/native/darwin/*.cc)
  FLAGS+=(
    "-I${JAVA_HOME}/include/darwin"
    "-Wl,-framework,CoreServices"
    "-Wl,-framework,IOKit"
  )
  ;;
openbsd)
  SOURCES+=(src/main/native/unix_jni_bsd.cc)
  FLAGS+=("-I${JAVA_HOME}/include/openbsd")
  ;;
freebsd)
  SOURCES+=(src/main/native/unix_jni_bsd.cc)
  FLAGS+=("-I${JAVA_HOME}/include/freebsd")
  ;;
esac

c++ "${FLAGS[@]}" "${SOURCES[@]}" -o "$OUT"
