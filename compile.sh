#!/bin/bash
set -o errexit
mkdir -p output/classes
mkdir -p output/test_classes
mkdir -p output/src
mkdir -p output/objs
mkdir -p output/native


# Compile .proto files using protoc
PROTO_FILES=(
src/main/protobuf/build.proto
src/main/protobuf/extra_actions.proto
src/main/protobuf/testing_api.proto
)

# TODO: CC target architecture needs to match JAVA_HOME.

# JAVA_HOME must point to a Java 7 installation.
JAVA_HOME=${JAVA_HOME:-$(readlink -f $(which javac) | sed "s_/bin/javac__")}
JAVAC="${JAVA_HOME}/bin/javac"
PROTOC=/usr/bin/protoc
CC=g++

for FILE in "${PROTO_FILES[@]}"; do
  echo "PROTOC ${FILE}"
  "${PROTOC}" \
      -Isrc/main/protobuf/ \
      --java_out=output/src \
      "${FILE}"
done

# Compile .java files (incl. generated ones) using javac
echo "JAVAC src/main/java/**/*.java"
CLASSPATH=third_party/guava/guava-16.0.1.jar:third_party/jsr305/jsr-305.jar:third_party/protobuf/protobuf-2.5.0.jar:third_party/joda-time/joda-time-2.3.jar
find src/main/java -name "*.java" | xargs "${JAVAC}" -classpath ${CLASSPATH} -sourcepath src/main/java:output/src -d output/classes

echo "UNZIP third_party/{guava,joda-time,jsr305,protobuf}/*.jar"
unzip -qn third_party/guava/guava-16.0.1.jar -d output/classes
unzip -qn third_party/joda-time/joda-time-2.3.jar -d output/classes
unzip -qn third_party/jsr305/jsr-305.jar -d output/classes
unzip -qn third_party/protobuf/protobuf-2.5.0.jar -d output/classes

# help files.
cp src/main/java/com/google/devtools/build/lib/blaze/commands/*.txt output/classes/com/google/devtools/build/lib/blaze/commands/

echo "JAR libblaze.jar"
echo "Main-Class: com.google.devtools.build.lib.blaze.main.BlazeMain" > output/MANIFEST.MF
jar cmf output/MANIFEST.MF output/libblaze.jar -C output/classes com/ -C output/classes javax/ -C output/classes org/

echo "JAVAC src/test/java/**/*.java"
find src/test/java -name "*.java" | xargs "${JAVAC}" -classpath ${CLASSPATH}:third_party/junit/junit-4.11.jar:third_party/truth/truth-0.20.jar:third_party/guava/guava-testlib.jar:output/classes -d output/test_classes

# Compile client .cc files.
CC_FILES=(
src/main/cpp/blaze_startup_options.cc
src/main/cpp/blaze_startup_options_common.cc
src/main/cpp/blaze_util.cc
src/main/cpp/blaze.cc
src/main/cpp/option_processor.cc
src/main/cpp/util/port.cc
src/main/cpp/util/strings.cc
src/main/cpp/util/file.cc
src/main/cpp/util/md5.cc
src/main/cpp/util/numbers.cc
)

for FILE in "${CC_FILES[@]}"; do
  if [[ ! "${FILE}" =~ ^-.*$ ]]; then
    echo "CC ${FILE}"
    OUT=$(basename "${FILE}").o
    "${CC}" \
        -I src/main/cpp/ \
        -I /usr/include/ \
        -std=c++0x \
        -c \
        -DBLAZE_JAVA_CPU=\"k8\" \
        -DBLAZE_OPENSOURCE=1 \
        -o "output/objs/${OUT}" \
        "${FILE}"
  fi
done

# Link client
echo "LD client"
"${CC}" -o output/client output/objs/*.o /usr/lib/x86_64-linux-gnu/libarchive.so -l stdc++ -l rt

# Compile native code .cc files.
NATIVE_CC_FILES=(
src/main/native/localsocket.cc
src/main/native/process.cc
src/main/native/unix_jni.cc
src/main/cpp/util/md5.cc
)

for FILE in "${NATIVE_CC_FILES[@]}"; do
  echo "CC ${FILE}"
  OUT=$(basename "${FILE}").o
  "${CC}" \
      -I src/main/cpp/ \
      -I src/main/native/ \
      -I "${JAVA_HOME}/include/" \
      -I "${JAVA_HOME}/include/linux/" \
      -std=c++0x \
      -fPIC \
      -c \
      -DBLAZE_JAVA_CPU=\"k8\" \
      -DBLAZE_OPENSOURCE=1 \
      -o "output/native/${OUT}" \
      "${FILE}"
done

echo "LD libunix.so"
"${CC}" -o output/libunix.so -shared output/native/*.o -l stdc++

echo "CC build-runfiles"
"${CC}" -o output/build-runfiles -std=c++0x -l stdc++ -l rt src/main/tools/build-runfiles.cc

echo "CC process-wrapper"
"${CC}" -o output/process-wrapper src/main/tools/process-wrapper.c

touch output/alarm
chmod 755 output/alarm

touch output/client_info
chmod 755 output/client_info

TO_ZIP="libblaze.jar libunix.so build-runfiles process-wrapper alarm client_info"
(cd output/ ; cat client ${TO_ZIP} | md5sum | awk '{ print $1; }' > install_base_key)
(cd output/ ; zip package.zip ${TO_ZIP} install_base_key)
cat output/client output/package.zip > output/bazel
zip -qA output/bazel
chmod 755 output/bazel
