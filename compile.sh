#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

set -o errexit
mkdir -p output/classes
mkdir -p output/test_classes
mkdir -p output/src
mkdir -p output/objs
mkdir -p output/native

# May be passed in from outside.
CFLAGS="$CFLAGS"
LDFLAGS="$LDFLAGS"
ZIPOPTS="$ZIPOPTS"

# TODO: CC target architecture needs to match JAVA_HOME.
CC=${CC:-g++}

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
JNIPLATFORM="${PLATFORM}"
ARCHIVE_CFLAGS=""
# Extension for executables (.exe on Windows).
EXE_EXT=""

PROTO_FILES=$(ls src/main/protobuf/*.proto)
LIBRARY_JARS="third_party/guava/guava-16.0.1.jar third_party/jsr305/jsr-305.jar third_party/protobuf/protobuf-2.5.0.jar third_party/joda-time/joda-time-2.3.jar"
TEST_LIBRARY_JARS="$LIBRARY_JARS third_party/junit/junit-4.11.jar third_party/truth/truth-0.23.jar third_party/guava/guava-testlib.jar output/classes"
DIRS=$(echo src/{main/java,tools/xcode-common/java/com/google/devtools/build/xcode/{common,util}})
JAVA_SRCDIRS="src/main/java src/tools/xcode-common/java output/src"

PATHSEP=":"

function fail() {
  echo "$1" >&2
  exit 1
}

case "${PLATFORM}" in
linux)
  # Sorry, no static linking on linux for now.
  LDFLAGS="$(pkg-config libarchive --libs) -lrt $LDFLAGS"
  JNILIB="libunix.so"
  MD5SUM="md5sum"
  # JAVA_HOME must point to a Java 7 installation.
  JAVA_HOME=${JAVA_HOME:-$(readlink -f $(which javac) | sed "s_/bin/javac__")}
  PROTOC=${PROTOC:-third_party/protobuf/protoc.amd64}
  ;;
darwin)
  homebrew_header=$(ls -1 /usr/local/Cellar/libarchive/*/include/archive.h | head -n1)
  if [[ -e $homebrew_header ]]; then
    # For use with Homebrew.
    archive_dir=$(dirname $(dirname $homebrew_header))
    ARCHIVE_CFLAGS="-I${archive_dir}/include"
    LDFLAGS="-L${archive_dir}/lib -larchive $LDFLAGS"
  elif [[ -e /opt/local/include/archive.h ]]; then
    # For use with Macports.
    ARCHIVE_CFLAGS="-I/opt/local/include"
    # Link libarchive statically
    LDFLAGS="/opt/local/lib/libarchive.a /opt/local/lib/liblzo2.a \
             /opt/local/lib/liblzma.a /opt/local/lib/libcharset.a \
             /opt/local/lib/libbz2.a /opt/local/lib/libxml2.a \
             /opt/local/lib/libz.a /opt/local/lib/libiconv.a \
             $LDFLAGS"
  else
    echo "WARNING: Could not find libarchive installation, proceeding bravely."
  fi

  JNILIB="libunix.dylib"
  MD5SUM="md5"
  JAVA_HOME=${JAVA_HOME:-$(/usr/libexec/java_home -v 1.7+)}
  PROTOC=${PROTOC:-protoc}
  ;;
msys*|mingw*)
  # Use a simplified platform string.
  PLATFORM="mingw"
  JNIPLATFORM=win32
  MD5SUM="md5sum"
  EXE_EXT=".exe"
  PATHSEP=";"
  # Find the latest available version of the SDK.
  JAVA_HOME="${JAVA_HOME:-$(ls -d /c/Program\ Files/Java/jdk* | sort | tail -n 1)}"
  # Export all symbols undecorated ("func" instead of "func@num"). Another option
  # is to use --add-stdcall-alias, which exports both versions.
  JNI_LD_ARGS="-Wl,--kill-at"
  JNILIB="unix.dll"
  PROTOC=${PROTOC:-protoc}

  # Ensure that we are using the cygwin gcc, not the mingw64 gcc.
  ${CC} -v 2>&1 | grep "Target: .*mingw.*" > /dev/null &&
    fail "mingw gcc detected. Please set CC to point to the Cygwin gcc."
esac

test -z "$JAVA_HOME" && fail "JDK not found, please set $$JAVA_HOME."

JAVAC="${JAVA_HOME}/bin/javac"
JAR="${JAVA_HOME}/bin/jar"

CLASSPATH=${LIBRARY_JARS// /$PATHSEP}
CLASSPATH_TESTS=${TEST_LIBRARY_JARS// /$PATHSEP}
SOURCEPATH=${JAVA_SRCDIRS// /$PATHSEP}

BLAZE_CC_FILES=(
src/main/cpp/blaze_startup_options.cc
src/main/cpp/blaze_startup_options_common.cc
src/main/cpp/blaze_util.cc
src/main/cpp/blaze_util_${PLATFORM}.cc
src/main/cpp/blaze.cc
src/main/cpp/option_processor.cc
src/main/cpp/util/port.cc
src/main/cpp/util/strings.cc
src/main/cpp/util/file.cc
src/main/cpp/util/md5.cc
src/main/cpp/util/numbers.cc
)

NATIVE_CC_FILES=(
src/main/cpp/util/md5.cc
src/main/native/localsocket.cc
src/main/native/process.cc
src/main/native/unix_jni.cc
src/main/native/unix_jni_${PLATFORM}.cc
)

if [ -z "${BAZEL_SKIP_JAVA_COMPILATION}" ]; then
  echo "Compiling Java stubs for protocol buffers..."
  for f in $PROTO_FILES ; do
    "${PROTOC}" -Isrc/main/protobuf/ --java_out=output/src "$f"
  done

  tmp="$(mktemp -d /tmp/bazel.XXXXXXXX)"
  paramfile="${tmp}.param"
  errfile="${tmp}.err"

  # Compile .java files (incl. generated ones) using javac
  echo "Compiling Bazel Java code..."
  find ${DIRS} -name "*.java" > "$paramfile"
  "${JAVAC}" -classpath "${CLASSPATH}" -sourcepath "$SOURCEPATH" \
      -d "output/classes" "@${paramfile}" &> "$errfile" ||
      { cat "$errfile" ; rm -f "$paramfile" "$errfile" ; exit 1 ; }
  rm "$paramfile" "$errfile"

  echo "Extracting helper classes..."
  for f in $LIBRARY_JARS ; do
    unzip -qn ${f} -d output/classes
  done

  # help files.
  cp src/main/java/com/google/devtools/build/lib/blaze/commands/*.txt \
      output/classes/com/google/devtools/build/lib/blaze/commands/

  echo "Creating libblaze.jar..."
  echo "Main-Class: com.google.devtools.build.lib.bazel.BazelMain" > output/MANIFEST.MF
  "$JAR" cmf output/MANIFEST.MF output/libblaze.jar \
      -C output/classes com/ -C output/classes javax/ -C output/classes org/

  echo "Compiling Bazel tests..."
  find src/test/java -name "*.java" > "$paramfile"
  "${JAVAC}" -classpath "${CLASSPATH_TESTS}" \
      -d "output/test_classes" "@${paramfile}" &> "$errfile" ||
      { cat "$errfile" ; rm -f "$paramfile" "$errfile" ; exit 1 ; }
  rm "$paramfile" "$errfile"
fi

echo "Compiling client .cc files..."
for FILE in "${BLAZE_CC_FILES[@]}"; do
  if [[ ! "${FILE}" =~ ^-.*$ ]]; then
    OUT=$(basename "${FILE}").o
    "${CC}" \
        -I src/main/cpp/ \
        ${ARCHIVE_CFLAGS} \
        ${CFLAGS} \
        -std=c++0x \
        -c \
        -DBLAZE_JAVA_CPU=\"k8\" \
        -DBLAZE_OPENSOURCE=1 \
        -o "output/objs/${OUT}" \
        "${FILE}"
  fi
done

# Link client
echo "Linking client..."
"${CC}" -o output/client output/objs/*.o -lstdc++ -larchive ${LDFLAGS}

echo "Compiling JNI libraries..."
for FILE in "${NATIVE_CC_FILES[@]}"; do
  OUT=$(basename "${FILE}").o
  "${CC}" \
      -I src/main/cpp/ \
      -I src/main/native/ \
      -I "${JAVA_HOME}/include/" \
      -I "${JAVA_HOME}/include/${JNIPLATFORM}" \
      -std=c++0x \
      -fPIC \
      -c \
      -D_JNI_IMPLEMENTATION_ \
      -DBLAZE_JAVA_CPU=\"k8\" \
      -DBLAZE_OPENSOURCE=1 \
      -o "output/native/${OUT}" \
      "${FILE}"
done

echo "Linking ${JNILIB}..."
"${CC}" -o output/${JNILIB} $JNI_LD_ARGS -shared output/native/*.o -l stdc++

echo "Compiling build-runfiles..."
# Clang on Linux requires libstdc++
"${CC}" -o output/build-runfiles -std=c++0x -l stdc++ src/main/tools/build-runfiles.cc

echo "Compiling process-wrapper..."
"${CC}" -o output/process-wrapper src/main/tools/process-wrapper.c

cp src/main/tools/build_interface_so output/build_interface_so

touch output/client_info
chmod 755 output/client_info

echo "Creating Bazel self-extracting archive..."
TO_ZIP="libblaze.jar ${JNILIB} build-runfiles${EXE_EXT} process-wrapper${EXE_EXT} client_info build_interface_so"
(cd output/ ; cat client ${TO_ZIP} | ${MD5SUM} | awk '{ print $1; }' > install_base_key)
(cd output/ ; zip $ZIPOPTS -q package.zip ${TO_ZIP} install_base_key)
cat output/client output/package.zip > output/bazel
zip -qA output/bazel || true  # Not all zip implementations provide -qA, and this step is optional anyway.

chmod 755 output/bazel
echo "Build successful!"
