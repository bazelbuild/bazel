#!/usr/bin/env bash

# Copyright 2018 The Bazel Authors. All rights reserved.
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

# This script creates from the full JDK a minimized version that only contains
# the specified JDK modules.

# --- begin runfiles.bash initialization v3 ---
# Copy-pasted from the Bazel Bash runfiles library v3.
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
# shellcheck disable=SC1090
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

# Force a UTF-8 compatible locale for Java tools to operate under paths with
# Unicode characters.
if [[ $(locale charmap) != "UTF-8" ]]; then
  export LC_CTYPE=C.UTF-8
fi
if [[ $(locale charmap) != "UTF-8" ]]; then
  export LC_CTYPE=en_US.UTF-8
fi

if [ "$1" == "--allmodules" ]; then
  shift
  modules="ALL-MODULE-PATH"
else
  modules=$(cat "$3" | paste -sd "," - | tr -d '\r')
  # We have to add this module explicitly because jdeps doesn't find the
  # dependency on it but it is still necessary for TLSv1.3.
  modules="$modules,jdk.crypto.ec"
fi
tooljdk=$1
fulljdk=$2
out=$4

UNAME=$(uname -s | tr 'A-Z' 'a-z')
# Options for the JVM that runs the Bazel server, which are either required or
# recommended when using the embedded JDK on platforms that use a minified JDK.
# Setting these options here rather than in blaze.cc avoids the need to detect
# compatible JDKs.
# Native access is required for the JNI library.
# Compact object headers reduce retained and peak memory usage.
JVM_OPTIONS='--enable-native-access=ALL-UNNAMED -XX:+UnlockExperimentalVMOptions -XX:+UseCompactObjectHeaders'

if [[ "$UNAME" =~ msys_nt* ]]; then
  mkdir "tmp.$$"
  cd "tmp.$$"
  unzip -q "../$fulljdk"
  cd zulu*
  # We have to add this module explicitly because it is windows specific, it allows
  # the usage of the Windows truststore
  # e.g. -Djavax.net.ssl.trustStoreType=WINDOWS-ROOT
  modules="$modules,jdk.crypto.mscapi"
  ./bin/jlink --module-path ./jmods/ --add-modules "$modules" \
    --vm=server --strip-debug --no-man-pages \
    --add-options=" ${JVM_OPTIONS}"\
    --output reduced
  # Patch the app manifest of the java.exe launcher to force its active code
  # page to UTF-8 on Windows 1903 and later, which is required for proper
  # support of Unicode characters outside the system code page.
  # The JDK currently (as of JDK 23) doesn't support this natively:
  # https://mail.openjdk.org/pipermail/core-libs-dev/2024-November/133773.html
  "$(rlocation io_bazel/src/read_manifest.exe)" reduced/bin/java.exe \
    | sed 's|</asmv3:windowsSettings>|<activeCodePage xmlns="http://schemas.microsoft.com/SMI/2019/WindowsSettings">UTF-8</activeCodePage>&|' \
    | "$(rlocation io_bazel/src/write_manifest.exe)" reduced/bin/java.exe
  cp DISCLAIMER readme.txt legal/java.base/ASSEMBLY_EXCEPTION \
    reduced/
  # These are necessary for --host_jvm_debug to work.
  cp bin/dt_socket.dll bin/jdwp.dll reduced/bin
  zip -q -X -r ../reduced.zip reduced/
  cd ../..
  mv "tmp.$$/reduced.zip" "$out"
  rm -rf "tmp.$$"
else
  # The --no-same-owner flag instructs tar to not try to chown extracted files
  # to the owner stored in the archive - it will try to do that when running as
  # root, but fail when running inside Docker, so we explicitly disable it.
  mkdir tool_jdk
  tar xf "$tooljdk" --no-same-owner --strip-components=1 -C tool_jdk
  mkdir target_jdk
  tar xf "$fulljdk" --no-same-owner --strip-components=1 -C target_jdk
  cd target_jdk
  "../tool_jdk/bin/jlink" --module-path ./jmods/ --add-modules "$modules" \
    --vm=server --strip-debug --no-man-pages \
    --add-options=" ${JVM_OPTIONS}" \
    --output reduced
  for f in DISCLAIMER readme.txt legal/java.base/ASSEMBLY_EXCEPTION; do [ -f "$f" ] && cp "$f" reduced/; done
  # These are necessary for --host_jvm_debug to work.
  cp lib/libdt_socket.* lib/libjdwp.* reduced/lib
  find reduced -exec touch -ht 198001010000 {} +
  zip -q -X -r ../reduced.zip reduced/
  cd ..
  mv reduced.zip "$out"
fi
