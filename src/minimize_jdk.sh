#!/usr/bin/env bash

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

# This script creates from the full JDK a minimized version that only contains
# the specified JDK modules. It runs on any host and can produce a minimized JDK
# for any target platform, including Windows.

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
# Optional 5th argument: a separate jmods archive for JDKs that don't ship
# with jmods (e.g. Adoptium Temurin with JEP 493 enabled).
jmods_archive=${5:-}
# Convert to absolute path since we cd later.
if [ -n "$jmods_archive" ]; then
  jmods_archive=$(cd "$(dirname "$jmods_archive")" && echo "$(pwd)/$(basename "$jmods_archive")")
fi

UNAME=$(uname -s | tr 'A-Z' 'a-z')
# Options for the JVM that runs the Bazel server, which are either required or
# recommended when using the embedded JDK on platforms that use a minified JDK.
# Setting these options here rather than in blaze.cc avoids the need to detect
# compatible JDKs.
# Native access is required for the JNI library.
# Compact object headers reduce retained and peak memory usage.
JVM_OPTIONS='--enable-native-access=ALL-UNNAMED -XX:+UseCompactObjectHeaders'
# Preserve heap shrinking after System.gc(), which Bazel uses to release memory
# when idle and after clean. JDK 27 changed G1's defaults to 0 and 100, disabling
# resizing after a full GC (JDK-8238686). Explicit --host_jvm_args override these
# defaults embedded by jlink.
JVM_OPTIONS+=' -XX:MinHeapFreeRatio=40 -XX:MaxHeapFreeRatio=70'

# Archives aren't necessarily named after their format (e.g. the tool JDK is
# copied to an extensionless file), so detect zip files by their content.
is_zip() {
  unzip -l "$1" >/dev/null 2>&1
}

# Extracts a JDK archive, which is a zip file for Windows JDKs and a tarball
# otherwise, into the given directory.
extract_jdk() {
  local archive=$1
  local dir=$2
  mkdir -p "$dir"
  if is_zip "$archive"; then
    unzip -q "$archive" -d "$dir"
  else
    # The --no-same-owner flag instructs tar to not try to chown extracted
    # files to the owner stored in the archive - it will try to do that when
    # running as root, but fail when running inside Docker, so we explicitly
    # disable it.
    tar xf "$archive" --no-same-owner -C "$dir"
  fi
}

# Strips an extracted JDK archive to its home directory.
# Some JDK archives (e.g., macOS .jdk bundles) nest the JDK home inside
# additional directories like Contents/Home/. This function detects the
# actual JDK home by locating the 'release' file and moves it to the
# expected top-level directory.
strip_to_jdk_home() {
  local dir=$1
  local release
  release=$(find "$dir" -name release -maxdepth 4 -print -quit)
  if [[ -z "$release" ]]; then
    echo "ERROR: could not find 'release' file in $dir" >&2
    exit 1
  fi
  local jdk_home
  jdk_home=$(dirname "$release")
  if [[ "$jdk_home" != "$dir" ]]; then
    local tmp="${dir}_tmp"
    mv "$jdk_home" "$tmp"
    rm -rf "$dir"
    mv "$tmp" "$dir"
  fi
}

extract_jdk "$tooljdk" "tool_jdk.$$"
strip_to_jdk_home "tool_jdk.$$"
tool_jdk_home=$(cd "tool_jdk.$$" && pwd)
extract_jdk "$fulljdk" "target_jdk.$$"
strip_to_jdk_home "target_jdk.$$"
cd "target_jdk.$$"

# The tool JDK runs on the host, the target JDK may be for another platform.
if [[ -f "$tool_jdk_home/bin/jlink.exe" ]]; then
  jlink="$tool_jdk_home/bin/jlink.exe"
else
  jlink="$tool_jdk_home/bin/jlink"
fi
if [[ -f bin/java.exe ]]; then
  target_windows=true
else
  target_windows=false
fi

# If the full JDK doesn't ship with jmods (e.g. JEP 493), use the separately
# provided jmods archive.
if [ ! -f jmods/java.base.jmod ]; then
  if [ -n "$jmods_archive" ]; then
    mkdir -p jmods
    if is_zip "$jmods_archive"; then
      unzip -q "$jmods_archive" -d jmods_tmp
      # The archive contains a single top-level directory with jmod files.
      mv jmods_tmp/*/* jmods_tmp/ 2>/dev/null || true
      mv jmods_tmp/*.jmod jmods/
      rm -rf jmods_tmp
    else
      tar xf "$jmods_archive" --no-same-owner --strip-components=1 -C jmods --wildcards '*.jmod'
    fi
  else
    echo >&2 "ERROR: Full JDK does not contain jmods/java.base.jmod and no" \
      "separate jmods archive was provided. Cross-jlinking requires jmods." \
      "JDKs with JEP 493 enabled (e.g. Adoptium Temurin 24+) need a separate" \
      "jmods download."
    exit 1
  fi
fi

if [[ "$target_windows" == true ]]; then
  # We have to add this module explicitly because it is windows specific, it allows
  # the usage of the Windows truststore
  # e.g. -Djavax.net.ssl.trustStoreType=WINDOWS-ROOT
  modules="$modules,jdk.crypto.mscapi"
fi

"$jlink" --module-path ./jmods/ --add-modules "$modules" \
  --vm=server --strip-debug --no-man-pages --no-header-files \
  --add-options=" ${JVM_OPTIONS}" \
  --output reduced

# Remove distrusted root CA certificates from the embedded JDK truststore
# (Sectigo AAA Certificate Services cross-sign root & eMudhra roots) to address
# the eMudhra intermediate CA private key compromise
# (see https://emudhra.we-have-a-problem.com/).
# Note: This must run AFTER jlink because jlink populates
# reduced/lib/security/cacerts from jmods/java.base.jmod.
if [[ -f "$tool_jdk_home/bin/keytool.exe" ]]; then
  keytool="$tool_jdk_home/bin/keytool.exe"
else
  keytool="$tool_jdk_home/bin/keytool"
fi
DISTRUSTED_SHA256=(
  "D7:A7:A0:FB:5D:7E:27:31:D7:71:E9:48:4E:BC:DE:F7:1D:5F:0C:3E:0A:29:48:78:2B:C8:3E:E0:EA:69:9E:F4" # CN=AAA Certificate Services
  "86:A1:EC:BA:08:9C:4A:8D:3B:BE:27:34:C6:12:BA:34:1D:81:3E:04:3C:F9:E8:A8:62:CD:5C:57:A3:6B:BE:6B" # CN=emSign ECC Root CA - G3
  "40:F6:AF:03:46:A9:9A:A1:CD:1D:55:5A:4E:9C:CE:62:C7:F9:63:46:03:EE:40:66:15:83:3D:C8:C8:D0:03:67" # CN=emSign Root CA - G1
  "1A:A0:C2:70:9E:83:1B:D6:E3:B5:12:9A:00:BA:41:F7:EE:EF:02:08:72:F1:E6:50:4B:F0:F6:C3:F2:4F:3A:F3" # CN=emSign Root CA - G2
  "12:56:09:AA:30:1D:A0:A2:49:B9:7A:82:39:CB:6A:34:21:6F:44:DC:AC:9F:39:54:B1:42:92:F2:E8:C8:60:8F" # CN=emSign Root CA - C1
  "BC:4D:80:9B:15:18:9D:78:DB:3E:1D:8C:F4:F9:72:6A:79:5D:A1:64:3C:A5:F1:35:8E:1D:DB:0E:DC:0D:7E:B3" # CN=emSign ECC Root CA - C3
)
keytool_list=$("$keytool" -list -v -keystore reduced/lib/security/cacerts -storepass changeit 2>/dev/null)
for fp in "${DISTRUSTED_SHA256[@]}"; do
  alias=$(echo "$keytool_list" | awk -v fp="$fp" '
    /^Alias name:/ { sub(/^Alias name: */, ""); sub(/\r$/, ""); a=$0 }
    index($0, "SHA256: " fp) > 0 { print a }
  ')
  if [[ -n "$alias" ]]; then
    "$keytool" -delete -alias "$alias" -keystore reduced/lib/security/cacerts -storepass changeit
  fi
done

if [[ "$target_windows" == true ]]; then
  # Patch the app manifest of the java.exe launcher to force its active code
  # page to UTF-8 on Windows 1903 and later, which is required for proper
  # support of Unicode characters outside the system code page.
  # The JDK currently (as of JDK 23) doesn't support this natively:
  # https://mail.openjdk.org/pipermail/core-libs-dev/2024-November/133773.html
  if [[ "$UNAME" =~ msys_nt* ]]; then
    "$(rlocation io_bazel/src/read_manifest.exe)" reduced/bin/java.exe \
      | sed 's|</asmv3:windowsSettings>|<activeCodePage xmlns="http://schemas.microsoft.com/SMI/2019/WindowsSettings">UTF-8</activeCodePage>&|' \
      | "$(rlocation io_bazel/src/write_manifest.exe)" reduced/bin/java.exe
  else
    echo >&2 "WARNING: not patching the app manifest of java.exe: only" \
      "supported when building on Windows"
  fi
fi
for f in DISCLAIMER readme.txt legal/java.base/ASSEMBLY_EXCEPTION; do [ -f "$f" ] && cp "$f" reduced/; done
# These are necessary for --host_jvm_debug to work.
if [[ "$target_windows" == true ]]; then
  cp bin/dt_socket.dll bin/jdwp.dll reduced/bin
else
  cp lib/libdt_socket.* lib/libjdwp.* reduced/lib
fi
find reduced -exec touch -ht 198001010000 {} +
zip -q -X -r ../reduced.zip reduced/
cd ..
mv reduced.zip "$out"
rm -rf "target_jdk.$$" "tool_jdk.$$"
