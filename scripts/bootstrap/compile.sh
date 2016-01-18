#!/bin/bash

# Copyright 2015 The Bazel Authors. All rights reserved.
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

# Script for building bazel from scratch without bazel

PROTO_FILES=$(ls src/main/protobuf/*.proto)
LIBRARY_JARS=$(find third_party -name '*.jar' | tr "\n" " ")
DIRS=$(echo src/{java_tools/singlejar/java/com/google/devtools/build/zip,main/java,tools/xcode-common/java/com/google/devtools/build/xcode/{common,util}} ${OUTPUT_DIR}/src)

case "${PLATFORM}" in
msys*|mingw*)
  BLAZE_UTIL_SUFFIX=mingw
  BLAZE_UTIL_POSIX=""
  ;;
*)
  BLAZE_UTIL_SUFFIX="${PLATFORM}"
  BLAZE_UTIL_POSIX="src/main/cpp/blaze_util_posix.cc"
  ;;
esac

BLAZE_CC_FILES=(
src/main/cpp/blaze_abrupt_exit.cc
src/main/cpp/blaze_startup_options.cc
src/main/cpp/blaze_startup_options_common.cc
src/main/cpp/blaze_util.cc
src/main/cpp/blaze_util_${BLAZE_UTIL_SUFFIX}.cc
${BLAZE_UTIL_POSIX}
src/main/cpp/blaze.cc
src/main/cpp/option_processor.cc
src/main/cpp/util/errors.cc
src/main/cpp/util/file.cc
src/main/cpp/util/md5.cc
src/main/cpp/util/numbers.cc
src/main/cpp/util/port.cc
src/main/cpp/util/strings.cc
third_party/ijar/zip.cc
)

NATIVE_CC_FILES=(
src/main/cpp/util/md5.cc
src/main/native/localsocket.cc
src/main/native/process.cc
src/main/native/unix_jni.cc
src/main/native/unix_jni_${PLATFORM}.cc
)

mkdir -p ${OUTPUT_DIR}/classes
mkdir -p ${OUTPUT_DIR}/test_classes
mkdir -p ${OUTPUT_DIR}/src
mkdir -p ${OUTPUT_DIR}/objs
mkdir -p ${OUTPUT_DIR}/native

# May be passed in from outside.
CXXFLAGS="$CXXFLAGS"
LDFLAGS="$LDFLAGS"
ZIPOPTS="$ZIPOPTS"

# TODO: CC target architecture needs to match JAVA_HOME.
if [ "${PLATFORM}" = "freebsd" ]; then
  CC=${CC:-clang}
  CXX=${CXX:-clang++}
else
  CC=${CC:-gcc}
  CXX=${CXX:-g++}
fi
CXXSTD="c++0x"

unset JAVA_TOOL_OPTIONS
unset _JAVA_OPTIONS

LDFLAGS=${LDFLAGS:-""}

# Extension for executables (.exe on Windows).
EXE_EXT=""

MSYS_DLLS=""
PATHSEP=":"

case "${PLATFORM}" in
linux)
  LDFLAGS="-lz -lrt $LDFLAGS"
  JNILIB="libunix.so"
  MD5SUM="md5sum"
  # JAVA_HOME must point to a Java installation.
  JAVA_HOME="${JAVA_HOME:-$(readlink -f $(which javac) | sed 's_/bin/javac__')}"
  if [ "${MACHINE_IS_64BIT}" = 'yes' ]; then
    PROTOC=${PROTOC:-third_party/protobuf/protoc-linux-x86_64.exe}
  else
    if [ "${MACHINE_IS_ARM}" = 'yes' ]; then
      PROTOC=${PROTOC:-third_party/protobuf/protoc-linux-arm32.exe}
    else
      PROTOC=${PROTOC:-third_party/protobuf/protoc-linux-x86_32.exe}
    fi
  fi
  ;;

freebsd)
  LDFLAGS="-lprocstat -lz -lrt $LDFLAGS"
  JNILIB="libunix.so"
  MD5SUM="md5"
  # JAVA_HOME must point to a Java installation.
  JAVA_HOME="${JAVA_HOME:-/usr/local/openjdk8}"
  # Note: the linux protoc binary works on freebsd using linux emulation.
  # We choose the 32-bit version for maximum compatiblity since 64-bit
  # linux binaries are only supported in FreeBSD-11.
  PROTOC=${PROTOC:-third_party/protobuf/protoc-linux-x86_32.exe}
  ;;

darwin)
  JNILIB="libunix.dylib"
  MD5SUM="md5"
  LDFLAGS="-lz $LDFLAGS"
  if [[ -z "$JAVA_HOME" ]]; then
    JAVA_HOME="$(/usr/libexec/java_home -v ${JAVA_VERSION}+ 2> /dev/null)" \
      || fail "Could not find JAVA_HOME, please ensure a JDK (version ${JAVA_VERSION}+) is installed."
  fi
  if [ "${MACHINE_IS_64BIT}" = 'yes' ]; then
    PROTOC=${PROTOC:-third_party/protobuf/protoc-osx-x86_64.exe}
  else
    PROTOC=${PROTOC:-third_party/protobuf/protoc-osx-x86_32.exe}
  fi
  ;;

msys*|mingw*)
  # Use a simplified platform string.
  PLATFORM="mingw"
  # Workaround for msys issue which causes omission of std::to_string.
  CXXFLAGS="$CXXFLAGS -D_GLIBCXX_USE_C99 -D_GLIBCXX_USE_C99_DYNAMIC"
  LDFLAGS="-lz $LDFLAGS"
  MD5SUM="md5sum"
  EXE_EXT=".exe"
  PATHSEP=";"
  # Find the latest available version of the SDK.
  JAVA_HOME="${JAVA_HOME:-$(ls -d /c/Program\ Files/Java/jdk* | sort | tail -n 1)}"
  # We do not use the JNI library on Windows.
  JNILIB=""
  if [ "${MACHINE_IS_64BIT}" = 'yes' ]; then
    PROTOC=${PROTOC:-third_party/protobuf/protoc-windows-x86_64.exe}
  else
    PROTOC=${PROTOC:-third_party/protobuf/protoc-windows-x86_32.exe}
  fi

  # The newer version of GCC on msys is stricter and removes some important function
  # declarations from the environment if using c++0x / c++11.
  CXXSTD="gnu++11"

  # Ensure that we are using the cygwin gcc, not the mingw64 gcc.
  ${CC} -v 2>&1 | grep "Target: .*mingw.*" > /dev/null &&
    fail "mingw gcc detected. Please set CC to point to the msys/Cygwin gcc."
  ${CXX} -v 2>&1 | grep "Target: .*mingw.*" > /dev/null &&
    fail "mingw g++ detected. Please set CXX to point to the msys/Cygwin g++."

  MSYS_DLLS="msys-2.0.dll msys-gcc_s-seh-1.dll msys-stdc++-6.dll"
  for dll in $MSYS_DLLS ; do
    cp "/usr/bin/$dll" "${OUTPUT_DIR}/$dll"
  done
esac

[[ -x "${PROTOC-}" ]] \
    || fail "Protobuf compiler not found in ${PROTOC-}"

# Check that javac -version returns a upper version than $JAVA_VERSION.
get_java_version
[ ${JAVA_VERSION#*.} -le ${JAVAC_VERSION#*.} ] || \
  fail "JDK version (${JAVAC_VERSION}) is lower than ${JAVA_VERSION}, please set \$JAVA_HOME."

JAR="${JAVA_HOME}/bin/jar"

# Compiles java classes.
function java_compilation() {
  local name=$1
  local directories=$2
  local library_jars=$3
  local output=$4

  local classpath=${library_jars// /$PATHSEP}:$5
  local sourcepath=${directories// /$PATHSEP}

  tempdir
  local tmp="${NEW_TMPDIR}"
  local paramfile="${tmp}/param"
  touch $paramfile

  mkdir -p "${output}/classes"

  # Compile .java files (incl. generated ones) using javac
  log "Compiling $name code..."
  find ${directories} -name "*.java" > "$paramfile"

  if [ ! -z "$BAZEL_DEBUG_JAVA_COMPILATION" ]; then
    echo "directories=${directories}" >&2
    echo "classpath=${classpath}" >&2
    echo "sourcepath=${sourcepath}" >&2
    echo "libraries=${library_jars}" >&2
    echo "output=${output}/classes" >&2
    echo "List of compiled files:" >&2
    cat "$paramfile" >&2
  fi

  run_silent "${JAVAC}" -classpath "${classpath}" -sourcepath "${sourcepath}" \
      -d "${output}/classes" -source "$JAVA_VERSION" -target "$JAVA_VERSION" \
      -encoding UTF-8 "@${paramfile}"

  log "Extracting helper classes for $name..."
  for f in ${library_jars} ; do
    run_silent unzip -qn ${f} -d "${output}/classes"
  done
}

# Create the deploy JAR
function create_deploy_jar() {
  local name=$1
  local mainClass=$2
  local output=$3
  shift 3
  local packages=""
  for i in $output/classes/*; do
    local package=$(basename $i)
    if [[ "$package" != "META-INF" ]]; then
      packages="$packages -C $output/classes $package"
    fi
  done

  log "Creating $name.jar..."
  echo "Main-Class: $mainClass" > $output/MANIFEST.MF
  run_silent "$JAR" cmf $output/MANIFEST.MF $output/$name.jar $packages "$@"
}

function cc_compile() {
  local OBJDIR=$1
  shift
  mkdir -p "${OUTPUT_DIR}/${OBJDIR}"
  for FILE in "$@"; do
    if [[ ! "${FILE}" =~ ^-.*$ ]]; then
      local OBJ=$(basename "${FILE}").o
      run_silent "${CXX}" \
          -I. \
          ${CFLAGS} \
          -std=$CXXSTD \
          -c \
          -DBLAZE_JAVA_CPU=\"k8\" \
          -DBLAZE_OPENSOURCE=1 \
          -o "${OUTPUT_DIR}/${OBJDIR}/${OBJ}" \
          "${FILE}"
    fi
  done
}

function cc_link() {
  local OBJDIR=$1
  local OUTPUT=$2
  shift 2
  local FILES=()
  for FILE in "$@"; do
    local OBJ=$(basename "${FILE}").o
    FILES+=("${OUTPUT_DIR}/${OBJDIR}/${OBJ}")
  done
  run_silent "${CXX}" -o ${OUTPUT} "${FILES[@]}" ${LDFLAGS}
}

function cc_build() {
  local NAME=$1
  local OBJDIR=$2
  local OUTPUT=$3
  shift 3
  log "Compiling ${NAME} .cc files..."
  cc_compile "${OBJDIR}" "$@"
  log "Linking ${NAME}..."
  cc_link "${OBJDIR}" "${OUTPUT}" "$@"
}

if [ -z "${BAZEL_SKIP_JAVA_COMPILATION}" ]; then
  log "Compiling Java stubs for protocol buffers..."
  for f in $PROTO_FILES ; do
    run_silent "${PROTOC}" -Isrc/main/protobuf/ --java_out=${OUTPUT_DIR}/src "$f"
  done

  java_compilation "Bazel Java" "$DIRS" "$LIBRARY_JARS" "${OUTPUT_DIR}"

  # help files: all non java and BUILD files in src/main/java.
  for i in $(find src/main/java -type f -a \! -name '*.java' -a \! -name 'BUILD' | sed 's|src/main/java/||'); do
    mkdir -p $(dirname ${OUTPUT_DIR}/classes/$i)
    cp src/main/java/$i ${OUTPUT_DIR}/classes/$i
  done

  # Overwrite tools.WORKSPACE, this is only for the bootstrap binary
  echo "local_repository(name = 'bazel_tools', path = __workspace_dir__)" \
       > ${OUTPUT_DIR}/classes/com/google/devtools/build/lib/bazel/rules/tools.WORKSPACE

  create_deploy_jar "libblaze" "com.google.devtools.build.lib.bazel.BazelMain" \
      ${OUTPUT_DIR}
fi

cc_build "client" "objs" "${OUTPUT_DIR}/client" ${BLAZE_CC_FILES[@]}

if [ ! -z "$JNILIB" ] ; then
  log "Compiling JNI libraries..."
  for FILE in "${NATIVE_CC_FILES[@]}"; do
    OUT=$(basename "${FILE}").o
    run_silent "${CXX}" \
      -I . \
      -I "${JAVA_HOME}/include/" \
      -I "${JAVA_HOME}/include/${PLATFORM}" \
      -std=$CXXSTD \
      -fPIC \
      -c \
      -D_JNI_IMPLEMENTATION_ \
      -DBLAZE_JAVA_CPU=\"k8\" \
      -DBLAZE_OPENSOURCE=1 \
      -o "${OUTPUT_DIR}/native/${OUT}" \
      "${FILE}"
  done

  log "Linking ${JNILIB}..."
  run_silent "${CXX}" -o ${OUTPUT_DIR}/${JNILIB} $JNI_LD_ARGS -shared ${OUTPUT_DIR}/native/*.o -lstdc++
fi

# Dummy build-runfiles
cat <<'EOF' >${OUTPUT_DIR}/build-runfiles${EXE_EXT}
#!/bin/bash
mkdir -p $2/MANIFEST
cp $1 $2/MANIFEST
EOF
chmod 0755 ${OUTPUT_DIR}/build-runfiles${EXE_EXT}

log "Compiling process-wrapper..."
cat <<'EOF' >${OUTPUT_DIR}/process-wrapper${EXE_EXT}
#!/bin/bash
# Dummy process wrapper, does not support timeout
shift 2
stdout="$1"
stderr="$2"
shift 2

"$@" 2>"$stderr" >"$stdout"
exit $?
EOF

log "Compiling xcode-locator..."
if [[ $PLATFORM == "darwin" ]]; then
  run_silent /usr/bin/xcrun clang -fobjc-arc -framework CoreServices -framework Foundation -o ${OUTPUT_DIR}/xcode-locator src/main/tools/xcode_locator.m
else
  cp src/main/tools/xcode_locator_stub.sh ${OUTPUT_DIR}/xcode-locator
fi

cp src/main/tools/build_interface_so ${OUTPUT_DIR}/build_interface_so
cp src/main/tools/jdk.* ${OUTPUT_DIR}

log "Creating Bazel self-extracting archive..."
ARCHIVE_DIR=${OUTPUT_DIR}/archive
for i in libblaze.jar ${JNILIB} build-runfiles${EXE_EXT} process-wrapper${EXE_EXT} xcode-locator${EXE_EXT} build_interface_so ${MSYS_DLLS} jdk.BUILD; do
  mkdir -p $(dirname $ARCHIVE_DIR/$i);
  cp $OUTPUT_DIR/$i $ARCHIVE_DIR/$i;
done

cp ${OUTPUT_DIR}/client ${ARCHIVE_DIR}
cp ${OUTPUT_DIR}/libblaze.jar ${ARCHIVE_DIR}/A-server.jar

(cd ${ARCHIVE_DIR}/ ; find . -type f | xargs cat | ${MD5SUM} | awk '{ print $1; }' > install_base_key)
(cd ${ARCHIVE_DIR}/ ; echo "${JAVA_VERSION}" > java.version)
(cd ${ARCHIVE_DIR}/ ; find . -type f | xargs -P 10 touch -t 198001010000)
(cd ${ARCHIVE_DIR}/ ; run_silent zip $ZIPOPTS -r -q package.zip * install_base_key java.version)
cat ${OUTPUT_DIR}/client ${ARCHIVE_DIR}/package.zip > ${OUTPUT_DIR}/bazel
zip -qA ${OUTPUT_DIR}/bazel \
  || echo "(Non-critical error, ignore.)"

chmod 755 ${OUTPUT_DIR}/bazel
