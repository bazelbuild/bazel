#!/bin/bash
#
# Copyright 2025 The Bazel Authors. All rights reserved.
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

case "$(uname -s | tr [:upper:] [:lower:])" in
linux*)
  declare -r PLATFORM=linux
  ;;
darwin*)
  declare -r PLATFORM=darwin
  ;;
msys*|mingw*|cygwin*)
  declare -r PLATFORM=windows
  ;;
*)
  declare -r PLATFORM=unknown
  ;;
esac

function is_linux() {
  [[ "$PLATFORM" == "linux" ]]
}

function is_darwin() {
  [[ "$PLATFORM" == "darwin" ]]
}

function is_windows() {
  [[ "$PLATFORM" == "windows" ]]
}

if is_windows; then
  declare -r EXE_EXT=".exe"
else
  declare -r EXE_EXT=""
fi
