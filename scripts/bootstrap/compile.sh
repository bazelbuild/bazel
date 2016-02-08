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

mkdir -p ${OUTPUT_DIR}/classes
mkdir -p ${OUTPUT_DIR}/src

# May be passed in from outside.
ZIPOPTS="$ZIPOPTS"

unset JAVA_TOOL_OPTIONS
unset _JAVA_OPTIONS

LDFLAGS=${LDFLAGS:-""}

# Extension for executables (.exe on Windows).
EXE_EXT=""

MSYS_DLLS=""
PATHSEP=":"

case "${PLATFORM}" in
linux)
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
  # JAVA_HOME must point to a Java installation.
  JAVA_HOME="${JAVA_HOME:-/usr/local/openjdk8}"
  # Note: the linux protoc binary works on freebsd using linux emulation.
  # We choose the 32-bit version for maximum compatiblity since 64-bit
  # linux binaries are only supported in FreeBSD-11.
  PROTOC=${PROTOC:-third_party/protobuf/protoc-linux-x86_32.exe}
  ;;

darwin)
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
  EXE_EXT=".exe"
  PATHSEP=";"
  # Find the latest available version of the SDK.
  JAVA_HOME="${JAVA_HOME:-$(ls -d /c/Program\ Files/Java/jdk* | sort | tail -n 1)}"
  # We do not use the JNI library on Windows.
  if [ "${MACHINE_IS_64BIT}" = 'yes' ]; then
    PROTOC=${PROTOC:-third_party/protobuf/protoc-windows-x86_64.exe}
  else
    PROTOC=${PROTOC:-third_party/protobuf/protoc-windows-x86_32.exe}
  fi
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

log "Creating Bazel install base..."
ARCHIVE_DIR=${OUTPUT_DIR}/archive
mkdir -p ${ARCHIVE_DIR}/_embedded_binaries

# Dummy build-runfiles
cat <<'EOF' >${ARCHIVE_DIR}/_embedded_binaries/build-runfiles${EXE_EXT}
#!/bin/bash
mkdir -p $2/MANIFEST
cp $1 $2/MANIFEST
EOF
chmod 0755 ${ARCHIVE_DIR}/_embedded_binaries/build-runfiles${EXE_EXT}

log "Creating process-wrapper..."
cat <<'EOF' >${ARCHIVE_DIR}/_embedded_binaries/process-wrapper${EXE_EXT}
#!/bin/bash
# Dummy process wrapper, does not support timeout
shift 2
stdout="$1"
stderr="$2"
shift 2

"$@" 2>"$stderr" >"$stdout"
exit $?
EOF
chmod 0755 ${ARCHIVE_DIR}/_embedded_binaries/process-wrapper${EXE_EXT}

cp src/main/tools/build_interface_so ${ARCHIVE_DIR}/_embedded_binaries/build_interface_so
cp src/main/tools/jdk.BUILD ${ARCHIVE_DIR}/_embedded_binaries/jdk.BUILD
cp $OUTPUT_DIR/libblaze.jar ${ARCHIVE_DIR}
cp src/main/tools/xcode_locator_stub.sh ${ARCHIVE_DIR}/_embedded_binaries/xcode-locator

# bazel build using bootstrap version
function bootstrap_build() {
  "${JAVA_HOME}/bin/java" \
      -client -Xms256m -XX:NewRatio=4 -XX:+HeapDumpOnOutOfMemoryError -Xverify:none -Dfile.encoding=ISO-8859-1 \
      -XX:HeapDumpPath=${OUTPUT_DIR} \
      -Djava.util.logging.config.file=${OUTPUT_DIR}/javalog.properties \
      -Dio.bazel.UnixFileSystem=0 \
      -jar ${ARCHIVE_DIR}/libblaze.jar \
      --batch \
      --install_base=${ARCHIVE_DIR} \
      --output_base=${OUTPUT_DIR}/out \
      --install_md5= \
      --workspace_directory=${PWD} \
      --nodeep_execroot --nofatal_event_bus_exceptions \
      build \
      --ignore_unsupported_sandboxing \
      --startup_time=329 --extract_data_time=523 \
      --rc_source=/dev/null --isatty=1 --terminal_columns=97 \
      --ignore_client_env \
      --client_cwd=${PWD} \
      "${@}"
}
