#!/usr/bin/env bash

# Copyright 2016 The Bazel Authors. All rights reserved.
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

# It's not a good idea to link an MSYS dynamic library into a native Windows
# JVM, so we need to build it with Visual Studio. However, Bazel doesn't
# support multiple compilers in the same build yet, so we need to hack around
# this limitation using a genrule.

set -eu

DLL="$1"
shift 1

function fail() {
  echo >&2 "ERROR: $*"
  exit 1
}

# Ensure the PATH is set up correctly.
if ! which which >&/dev/null ; then
  PATH="/bin:/usr/bin:$PATH"
  which which >&/dev/null \
      || fail "System PATH is not set up correctly, cannot run GNU bintools"
fi

# Create a temp directory. It will used for the batch file we generate soon and
# as the temp directory for CL.EXE .
VSTEMP=$(mktemp -d)
trap "rm -fr \"$VSTEMP\"" EXIT
VSVARS=""

# Visual Studio or Visual C++ Build Tools might not be installed at default
# location. Check BAZEL_VS and BAZEL_VC first.
if [ -n "${BAZEL_VC+set}" ]; then
  VSVARS="${BAZEL_VC}/VCVARSALL.BAT"
  # Check if BAZEL_VC points to Visual C++ Build Tools 2019
  if [ ! -f "${VSVARS}" ]; then
    VSVARS="${BAZEL_VC}/Auxiliary/Build/VCVARSALL.BAT"
  fi
else
  # Find Visual Studio. We don't have any regular environment variables
  # available so this is the best we can do.
  if [ -z "${BAZEL_VS+set}" ]; then
    VSVERSION="$(ls "C:/Program Files (x86)" \
        | grep -E "Microsoft Visual Studio [0-9]+" \
        | sort --version-sort \
        | tail -n 1)"
    BAZEL_VS="C:/Program Files (x86)/$VSVERSION"
  fi
  VSVARS="${BAZEL_VS}/VC/VCVARSALL.BAT"
fi

# Check if Visual Studio 2019 is installed. Look for it at the default
# locations.
if [ ! -f "${VSVARS}" ]; then
  VSVARS="C:/Program Files (x86)/Microsoft Visual Studio/2019/"
  VSEDITION="BuildTools"
  if [ -d "${VSVARS}Enterprise" ]; then
    VSEDITION="Enterprise"
  elif [ -d "${VSVARS}Professional" ]; then
    VSEDITION="Professional"
  elif [ -d "${VSVARS}Community" ]; then
    VSEDITION="Community"
  fi
  VSVARS+="$VSEDITION/VC/Auxiliary/Build/VCVARSALL.BAT"
fi

if [ ! -f "${VSVARS}" ]; then
  fail "VCVARSALL.bat not found, check your Visual Studio installation"
fi

JAVAINCLUDES=""
if [ -n "${JAVA_HOME+set}" ]; then
  JAVAINCLUDES="$JAVA_HOME/include"
else
  # Find Java. $(JAVA) in the BUILD file points to external/local_jdk/...,
  # which is not very useful for anything not MSYS-based.
  JAVA=$(ls "C:/Program Files/java" | grep -E "^jdk" | sort | tail -n 1)
  [[ -n "$JAVA" ]] || fail "JDK not found"
  JAVAINCLUDES="C:/Program Files/java/$JAVA/include"
fi

# Convert all compilation units to Windows paths.
WINDOWS_SOURCES=()
for i in $*; do
  if [[ "$i" =~ ^.*\.cc$ ]]; then
    WINDOWS_SOURCES+=("\"$(cygpath -a -w $i)\"")
  fi
done

# Copy jni headers to src/main/native folder
# Mimic genrule //src/main/native:copy_link_jni_md_header and //src/main/native:copy_link_jni_header
JNI_HEADERS_DIR="${VSTEMP}/src/main/native"
mkdir -p "$JNI_HEADERS_DIR"
cp -f "$JAVAINCLUDES/jni.h" "$JNI_HEADERS_DIR/"
cp -f "$JAVAINCLUDES/win32/jni_md.h" "$JNI_HEADERS_DIR/"

# CL.EXE needs a bunch of environment variables whose official location is a
# batch file. We can't make that have an effect on a bash instance, so
# generate a batch file that invokes it.
# As for `abs_pwd` and `pwd_drive`: in cmd.exe, it's not enough to `cd` into a
# directory. You must also change to its drive to truly set the cwd to that
# directory. See https://github.com/bazelbuild/bazel/issues/3906
abs_pwd="$(cygpath -a -w "${PWD}")"
pwd_drive="$(echo "$abs_pwd" | head -c2)"
cat > "${VSTEMP}/windows_jni.bat" <<EOF
@echo OFF
@call "${VSVARS}" amd64
@$pwd_drive
@cd "$abs_pwd"
@set TMP=$(cygpath -a -w "${VSTEMP}")
@CL /O2 /EHsc /LD /Fe:"$(cygpath -a -w ${DLL})" /I "%TMP%" /I . /I ${JNI_HEADERS_DIR} ${WINDOWS_SOURCES[*]} /link /DEFAULTLIB:advapi32.lib
EOF

# Invoke the file and hopefully generate the .DLL .
chmod +x "${VSTEMP}/windows_jni.bat"
exec "${VSTEMP}/windows_jni.bat"
