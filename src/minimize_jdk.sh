#!/bin/bash

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

set -euo pipefail

if [ "$1" == "--allmodules" ]; then
  shift
  modules="ALL-MODULE-PATH"
else
  modules=$(cat "$2" | paste -sd "," - | tr -d '\r')
  # We have to add this module explicitly because jdeps doesn't find the
  # dependency on it but it is still necessary for TLSv1.3.
  modules="$modules,jdk.crypto.ec"
fi
fulljdk=$1
out=$3
ARCH=`uname -p`
if [[ "${ARCH}" == 'ppc64le'  ]] || [[ "${ARCH}" == 's390x' ]]; then
  FULL_JDK_DIR="jdk*"
  DOCS=""
else
  FULL_JDK_DIR="zulu*"
  DOCS="DISCLAIMER readme.txt"
fi

UNAME=$(uname -s | tr 'A-Z' 'a-z')

# Remove distrusted root CA certificates from the embedded JDK truststore
# (Sectigo AAA Certificate Services cross-sign root & eMudhra roots) to address
# the eMudhra intermediate CA private key compromise
# (see https://emudhra.we-have-a-problem.com/).
# Note: This must run AFTER jlink because jlink populates
# reduced/lib/security/cacerts from jmods/java.base.jmod.
remove_distrusted_certs() {
  local DISTRUSTED_SHA256=(
    "D7:A7:A0:FB:5D:7E:27:31:D7:71:E9:48:4E:BC:DE:F7:1D:5F:0C:3E:0A:29:48:78:2B:C8:3E:E0:EA:69:9E:F4" # CN=AAA Certificate Services
    "86:A1:EC:BA:08:9C:4A:8D:3B:BE:27:34:C6:12:BA:34:1D:81:3E:04:3C:F9:E8:A8:62:CD:5C:57:A3:6B:BE:6B" # CN=emSign ECC Root CA - G3
    "40:F6:AF:03:46:A9:9A:A1:CD:1D:55:5A:4E:9C:CE:62:C7:F9:63:46:03:EE:40:66:15:83:3D:C8:C8:D0:03:67" # CN=emSign Root CA - G1
    "1A:A0:C2:70:9E:83:1B:D6:E3:B5:12:9A:00:BA:41:F7:EE:EF:02:08:72:F1:E6:50:4B:F0:F6:C3:F2:4F:3A:F3" # CN=emSign Root CA - G2
    "12:56:09:AA:30:1D:A0:A2:49:B9:7A:82:39:CB:6A:34:21:6F:44:DC:AC:9F:39:54:B1:42:92:F2:E8:C8:60:8F" # CN=emSign Root CA - C1
    "BC:4D:80:9B:15:18:9D:78:DB:3E:1D:8C:F4:F9:72:6A:79:5D:A1:64:3C:A5:F1:35:8E:1D:DB:0E:DC:0D:7E:B3" # CN=emSign ECC Root CA - C3
  )
  local keytool_list
  keytool_list=$(./bin/keytool -list -v -keystore reduced/lib/security/cacerts -storepass changeit 2>/dev/null)
  for fp in "${DISTRUSTED_SHA256[@]}"; do
    local alias
    alias=$(echo "$keytool_list" | awk -v fp="$fp" '
      /^Alias name:/ { sub(/^Alias name: */, ""); sub(/\r$/, ""); a=$0 }
      index($0, "SHA256: " fp) > 0 { print a }
    ')
    if [[ -n "$alias" ]]; then
      ./bin/keytool -delete -alias "$alias" -keystore reduced/lib/security/cacerts -storepass changeit
    fi
  done
}

if [[ "$UNAME" =~ msys_nt* ]]; then
  set -x
  mkdir "tmp.$$"
  cd "tmp.$$"
  unzip -q "../$fulljdk"
  cd $FULL_JDK_DIR
  # We have to add this module explicitly because it is windows specific, it allows
  # the usage of the Windows truststore
  # e.g. -Djavax.net.ssl.trustStoreType=WINDOWS-ROOT
  modules="$modules,jdk.crypto.mscapi"
  ./bin/jlink --module-path ./jmods/ --add-modules "$modules" \
    --vm=server --strip-debug --no-man-pages \
    --output reduced
  remove_distrusted_certs
  cp $DOCS legal/java.base/ASSEMBLY_EXCEPTION \
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
  tar xf "$fulljdk" --no-same-owner
  cd $FULL_JDK_DIR
  ./bin/jlink --module-path ./jmods/ --add-modules "$modules" \
    --vm=server --strip-debug --no-man-pages \
    --output reduced
  remove_distrusted_certs
  cp $DOCS legal/java.base/ASSEMBLY_EXCEPTION \
    reduced/
  # These are necessary for --host_jvm_debug to work.
  if [[ "$UNAME" =~ darwin ]]; then
    cp lib/libdt_socket.dylib lib/libjdwp.dylib reduced/lib
  else
    cp lib/libdt_socket.so lib/libjdwp.so reduced/lib
  fi
  find reduced -exec touch -ht 198001010000 {} +
  zip -q -X -r ../reduced.zip reduced/
  cd ..
  mv reduced.zip "$out"
fi
