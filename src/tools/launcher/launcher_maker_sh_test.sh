#!/usr/bin/env bash
#
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

# Verifies that tools/launcher/launcher_maker.sh produces the same output as
# launcher_maker.cc.
#
# Usage: launcher_maker_sh_test.sh <launcher_maker> <launcher_maker.sh>

set -euo pipefail

readonly CC_LAUNCHER_MAKER="$PWD/$1"
readonly SH_LAUNCHER_MAKER="$PWD/$2"
# Any binary file can serve as the launcher.
readonly LAUNCHER="$CC_LAUNCHER_MAKER"

cd "$TEST_TMPDIR"

failed=0
function check() {
  local -r name="$1"
  "$CC_LAUNCHER_MAKER" "$LAUNCHER" "$name.params" "$name.cc.exe"
  "$SH_LAUNCHER_MAKER" "$LAUNCHER" "$name.params" "$name.sh.exe"
  if ! cmp "$name.cc.exe" "$name.sh.exe"; then
    echo "FAILED: launcher_maker.sh output differs for $name" >&2
    failed=1
  fi
}

printf '%s\n' \
  'binary_type=Bash' \
  'workspace_name=_main' \
  'bash_bin_path=C:\msys64\usr\bin\bash.exe' \
  "$(printf 'list=a\tb\tc')" \
  'empty=' \
  '' \
  '  surrounding spaces  ' \
  '-n' \
  'format=%s %d \n \000' \
  "$(printf 'utf8=\303\274')" \
  >typical.params
printf 'crlf=1\r\n' >>typical.params
check typical

printf 'first=1\nlast=2' >no_trailing_newline.params
check no_trailing_newline

: >empty.params
check empty

# Requires three bytes to encode the length.
{
  printf 'long='
  head -c 70000 /dev/zero | tr '\0' x
  printf '\n'
} >long.params
check long

exit "$failed"
