#!/bin/bash -eu
#
# Copyright 2015 Google Inc. All rights reserved.
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

# TODO(bazel-team): this file is bloated, we should factor it.

#### Inputs

## TEST_TMPDIR
if [ -z "${TEST_TMPDIR:-}" ]; then
   TEST_TMPDIR="$(mktemp -d ${TMPDIR:-/tmp}/ijar-test.XXXXXXXX)"
   trap "rm -fr ${TEST_TMPDIR}" EXIT
fi

## Mac OS X stat and MD5
PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
if [[ "$PLATFORM" = "darwin" ]]; then
  function statfmt() {
    stat -f "%z" $1
  }
  MD5SUM=/sbin/md5
else
  function statfmt() {
    stat -c "%s" $1
  }
  MD5SUM=md5sum
fi

## Tools
JAVAC=$1
JAVA=$2
JAR=$3
JAVAP=$4
IJAR=$TEST_SRCDIR/$5
LANGTOOLS8=$TEST_SRCDIR/$6
UNZIP=$7
ZIP=$8

IJAR_SRCDIR=$(dirname ${IJAR})
A_JAR=$TEST_TMPDIR/A.jar
A_INTERFACE_JAR=$TEST_TMPDIR/A-interface.jar
A_ZIP_JAR=$TEST_TMPDIR/A_zip.jar
A_ZIP_INTERFACE_JAR=$TEST_TMPDIR/A_zip-interface.jar
W_JAR=$TEST_TMPDIR/W.jar
BOTTLES_JAR=$TEST_TMPDIR/bottles.jar
JAR_WRONG_CENTRAL_DIR=$IJAR_SRCDIR/test/libwrongcentraldir.jar
IJAR_WRONG_CENTRAL_DIR=$TEST_TMPDIR/wrongcentraldir_interface.jar
OBJECT_JAVA=$IJAR_SRCDIR/test/Object.java
OBJECT_JAR=$TEST_TMPDIR/object.jar
OBJECT_IJAR=$TEST_TMPDIR/object_interface.jar
TYPEANN2_JAR=$IJAR_SRCDIR/test/libtypeannotations2.jar
TYPEANN2_IJAR=$TEST_TMPDIR/typeannotations2_interface.jar
TYPEANN2_JAVA=$IJAR_SRCDIR/test/TypeAnnotationTest2.java
INVOKEDYNAMIC_JAR=$IJAR_SRCDIR/test/libinvokedynamic.jar
INVOKEDYNAMIC_IJAR=$TEST_TMPDIR/invokedynamic_interface.jar

#### Testing framework
# Print message in "$1" then exit with status "$2"
die () {
  # second argument is optional, defaulting to 1
  local status_code=${2:-1}
  # Stop capturing stdout/stderr, and dump captured output
  if [ "$CAPTURED_STD_ERR" -ne 0 ]; then
    restore_outputs
    cat "${TEST_TMPDIR}/captured.err" 1>&2
  fi

  if [ -n "${1-}" ] ; then
    echo "$1" 1>&2
  fi
  if [ x"$status_code" != x -a x"$status_code" != x"0" ]; then
    exit "$status_code"
  else
    exit 1
  fi
}

# Die if "$1" == "$2", print $3 as death reason
check_ne () {
  if [ "$1" = "$2" ]; then
    die "Check failed: '$1' != '$2' ${3:+ ($3)}"
  fi
}

# Die if "$1" != "$2", print $3 as death reason
check_eq () {
  if [ ! "$1" = "$2" ]; then
    die "Check failed: '$1' = '$2' ${3:+ ($3)}"
  fi
}

CAPTURED_STD_ERR="${CAPTURED_STD_ERR:-0}"

capture_test_stderr () {
  exec 6>&2 # Save stderr as fd 6
  exec 7>"${TEST_TMPDIR}/captured.err"
  exec 2>&7
  CAPTURED_STD_ERR=1
}

restore_outputs () {
  if [ "$CAPTURED_STD_ERR" -ne 0 ] ; then
    exec 2>&6
  fi
}

recapture_outputs () {
  if [ "$CAPTURED_STD_ERR" -ne 0 ] ; then
    exec 2>&7
  fi
}

#### Setup

# set_file_length FILE SIZE
#
# Sets the file size for FILE, truncating if necessary, creating a
# sparse file if possible, preserving original contents if they fit.
function set_file_length() {
  perl -e 'open(FH, ">>$ARGV[0]") && truncate(FH, $ARGV[1]) or die $!' "$@" &&
  [[ "$(statfmt $1)" == "$2" ]] ||
  die "set_file_length failed"
}

# grep_test_stderr STRING MESSAGE
#
# Greps the captured stderr text for STRING. If not found, fails with MESSAGE.
grep_test_stderr() {
  restore_outputs
  [ "$CAPTURED_STD_ERR" -ne 0 ] || \
    die "Must call capture_test_stderr before grep_test_stderr"
  grep -c "$1" ${TEST_TMPDIR}/captured.err >/dev/null || \
    die "Check failed: grep -c '$1' ${TEST_TMPDIR}/captured.err ${2:+ ($2)}"
  CAPTURED_STD_ERR=0
  recapture_outputs
}

# check_consistent_file_contents FILE
#
# Checks that all files created with the given filename have identical contents.
expected_output=""
function check_consistent_file_contents() {
  local actual="$(cat $1 | ${MD5SUM} | awk '{ print $1; }')"
  local filename="$(echo $1 | ${MD5SUM} | awk '{ print $1; }')"
  local expected="$actual"
  if $(echo "${expected_output}" | grep -q "^${filename} "); then
    echo "${expected_output}" | grep -q "^${filename} ${actual}$" || {
      ls -l "$1"
      die "output file contents differ"
    }
  else
    expected_output="$expected_output$filename $actual
"
  fi
}

# Tests that ijar does not crash when output ijar is bigger than the input jar
rm -fr $TEST_TMPDIR/classes
mkdir -p $TEST_TMPDIR/classes || die "mkdir $TEST_TMPDIR/classes failed"
$JAVAC -g -d $TEST_TMPDIR/classes \
    $IJAR_SRCDIR/test/WellCompressed*.java ||
    die "javac failed"
$JAR cf $W_JAR -C $TEST_TMPDIR/classes . || die "jar failed"

W_INTERFACE_JAR=$TEST_TMPDIR/W-interface.jar
$IJAR $W_JAR $W_INTERFACE_JAR || die "ijar failed"

mkdir -p $TEST_TMPDIR/java/lang
cp $OBJECT_JAVA $TEST_TMPDIR/java/lang/.
$JAVAC $TEST_TMPDIR/java/lang/Object.java || die "javac failed"
$JAR cf $OBJECT_JAR -C $TEST_TMPDIR java/lang/Object.class || die "jar failed"

# Tests that ijar can handle class bodies longer than 64K
rm -fr $TEST_TMPDIR/classes
mkdir -p $TEST_TMPDIR/classes || die "mkdir $TEST_TMPDIR/classes failed"
# First, generate the input file
BOTTLES_JAVA=$TEST_TMPDIR/BottlesOnTheWall.java
echo "public class BottlesOnTheWall {" > $BOTTLES_JAVA
for i in $(seq 1 16384); do
  echo "  public int getBottleOnTheWall${i}() { return ${i}; }" >> $BOTTLES_JAVA
done

echo "}" >> $BOTTLES_JAVA

$JAVAC -g -d $TEST_TMPDIR/classes $BOTTLES_JAVA || die "javac failed"
BOTTLES_INTERFACE_JAR=$TEST_TMPDIR/bottles-interface.jar

for flag0 in '' '0'; do
  $JAR c"${flag0}"f $BOTTLES_JAR -C $TEST_TMPDIR/classes . || die "jar failed"
  $IJAR $BOTTLES_JAR $BOTTLES_INTERFACE_JAR || die "ijar failed"
  check_consistent_file_contents $BOTTLES_INTERFACE_JAR
done

# Compiles A.java, builds A.jar and A-interface.jar
rm -fr $TEST_TMPDIR/classes
mkdir -p $TEST_TMPDIR/classes || die "mkdir $TEST_TMPDIR/classes failed"
$JAVAC -g -d $TEST_TMPDIR/classes $IJAR_SRCDIR/test/A.java ||
    die "javac failed"

for flag0 in '' '0'; do
# Ensure input files larger than INITIAL_BUFFER_SIZE work.
# TODO(martinrb): remove maximum .class file size limit (MAX_BUFFER_SIZE)
for size in '' $((1024*1024)) $((15*1024*1024)); do
    if [[ -n "$size" ]]; then
      for file in $(find $TEST_TMPDIR/classes -name '*.class'); do
        set_file_length "$file" "$size"
      done
    fi
    $JAR c"${flag0}"f $A_JAR -C $TEST_TMPDIR/classes . || die "jar failed"
    $IJAR $A_JAR $A_INTERFACE_JAR || die "ijar failed."
    check_consistent_file_contents $A_INTERFACE_JAR
  done
done

# Creates a huge (3Gb) input jar to test "large file" correctness
set_file_length $TEST_TMPDIR/zeroes.data $((3*1024*1024*1024))
for flag0 in '' '0'; do
  $JAR c"${flag0}"f $A_JAR -C $TEST_TMPDIR zeroes.data -C $TEST_TMPDIR/classes . || die "jar failed"
  $IJAR $A_JAR $A_INTERFACE_JAR || die "ijar failed."
  check_consistent_file_contents $A_INTERFACE_JAR
done

# Create an output jar with upper bound on size > 2GB
DIR=$TEST_TMPDIR/ManyLargeClasses
mkdir -p $DIR/classes
for i in $(seq 200); do
  printf "class C${i} {}\n" > $DIR/C${i}.java
done
([[ "$JAVAC" =~ ^/ ]] || JAVAC="$PWD/$JAVAC"; cd $DIR && $JAVAC -d classes *.java)
for i in $(seq 200); do
  set_file_length $DIR/classes/C${i}.class $((15*1024*1024))
done
$JAR cf $DIR/ManyLargeClasses.jar -C $DIR/classes . || die "jar failed"
$IJAR $DIR/ManyLargeClasses.jar $DIR/ManyLargeClasses.ijar || die "ijar failed."

#### Checks

# Check that ijar can produce class files with a body longer than 64K by
# calling ijar itself on the output file to make sure that it is valid
BOTTLES_INTERFACE_INTERFACE_JAR=$TEST_TMPDIR/bottles-interface-interface.jar
$IJAR $BOTTLES_INTERFACE_JAR $BOTTLES_INTERFACE_INTERFACE_JAR ||
    die "ijar cannot produce class files with body longer than 64K"

# Check that the interface jar is bigger than the original jar.
W_JAR_SIZE=$(statfmt $W_JAR)
W_INTERFACE_JAR_SIZE=$(statfmt $W_INTERFACE_JAR)
[[ $W_INTERFACE_JAR_SIZE -gt $W_JAR_SIZE ]] || die "interface jar should be bigger"

# Check that the number of entries is 5:
#  A, A.PrivateInner, A.PublicInner, A.MyAnnotation,
#  A.RuntimeAnnotation
# (Note: even private inner classes are retained, so we don't need to change
# the types of members.)
lines=$($JAR tvf $A_INTERFACE_JAR | wc -l)
expected=5
check_eq $expected $lines "Interface jar should have $expected entries!"


# Check that no private class members are found:
lines=$($JAVAP -private -classpath $A_JAR A | grep priv | wc -l)
check_eq 2 $lines "Input jar should have 2 private members!"
lines=$($JAVAP -private -classpath $A_INTERFACE_JAR A | grep priv | wc -l)
check_eq 0 $lines "Interface jar should have no private members!"
lines=$($JAVAP -private -classpath $A_INTERFACE_JAR A | grep clinit | wc -l)
check_eq 0 $lines "Interface jar should have no class initializers!"


# Check that no code is found:
lines=$($JAVAP -c -private -classpath $A_JAR A | grep Code: | wc -l)
check_eq 5 $lines "Input jar should have 5 method bodies!"
lines=$($JAVAP -c -private -classpath $A_INTERFACE_JAR A | grep Code: | wc -l)
check_eq 0 $lines "Interface jar should have no method bodies!"


# Check that constants from code are no longer present:
$JAVAP -c -private -classpath $A_JAR A | grep -sq foofoofoofoo ||
    die "Input jar should have code constants!"
$JAVAP -c -private -classpath $A_INTERFACE_JAR A | grep -sq foofoofoofoo &&
    die "Interface jar should have no code constants!"


# Check (important, this!) that the interface jar is still sufficient
# for compiling:
$JAVAC -Xlint -classpath $A_INTERFACE_JAR -g -d $TEST_TMPDIR/classes \
    $IJAR_SRCDIR/test/B.java 2>$TEST_TMPDIR/B.javac.err ||
    { cat $TEST_TMPDIR/B.javac.err >&2; die "Can't compile B!"; }


# Test compilation of B yielded deprecation message:
grep -sq 'deprecatedMethod.*in A has been deprecated' \
    $TEST_TMPDIR/B.javac.err || die "ijar has lost @Deprecated annotation!"


# Check idempotence of ijar transformation:
A_INTERFACE_INTERFACE_JAR=$TEST_TMPDIR/A-interface-interface.jar
$IJAR $A_INTERFACE_JAR $A_INTERFACE_INTERFACE_JAR || die "ijar failed."
cmp $A_INTERFACE_JAR $A_INTERFACE_INTERFACE_JAR ||
    die "ijar transformation is not idempotent"


# Check that -interface.jar contains nothing but .class files:
check_eq 0 $($JAR tf $A_INTERFACE_JAR | grep -v \\.class$ | wc -l) \
    "Interface jar should contain only .class files!"


# Check that -interface.jar timestamps are all zeros:
check_eq 0 $(TZ=UTC $JAR tvf $A_INTERFACE_JAR |
             grep -v 'Fri Nov 30 00:00:00 UTC 1979' | wc -l) \
    "Interface jar contained non-zero timestamps!"


# Check that compile-time constants in A are still annotated as such in ijar:
$JAVAP -classpath $TEST_TMPDIR/classes -c B | grep -sq ldc2_w.*123 ||
    die "ConstantValue not propagated to class B!"

# Regression test for jar file without classes (javac doesn't like an empty ijar).
>$TEST_TMPDIR/empty
$ZIP $TEST_TMPDIR/noclasses.jar $TEST_TMPDIR/empty >/dev/null 2>&1
$IJAR $TEST_TMPDIR/noclasses.jar || die "ijar failed"
$UNZIP -qql $TEST_TMPDIR/noclasses-interface.jar 2>/dev/null | grep -q . ||
    die "noclasses-interface.jar is completely empty!"
    $JAVAC -classpath $TEST_TMPDIR/noclasses-interface.jar \
    -d $TEST_TMPDIR/classes \
    $IJAR_SRCDIR/test/A.java ||
    die "javac noclasses-interface.jar failed"
rm $TEST_TMPDIR/{empty,noclasses.jar,noclasses-interface.jar}


# Run the dynamic checks in B.main().
$JAVA -classpath $TEST_TMPDIR/classes B || exit 1

# TODO(bazel-team) test that modifying the source in a non-interface
#   changing way results in the same -interface.jar.

# Check that a jar compressed with zip results in the same interface jar as a
# jar compressed with jar
rm -fr $TEST_TMPDIR/classes
mkdir -p $TEST_TMPDIR/classes || die "mkdir $TEST_TMPDIR/classes failed"
$JAVAC -g -d $TEST_TMPDIR/classes $IJAR_SRCDIR/test/A.java ||
    die "javac failed"
$JAR cf $A_JAR $TEST_TMPDIR/classes/A.class || die "jar failed"
$ZIP $A_ZIP_JAR $TEST_TMPDIR/classes/A.class || die "zip failed"

$IJAR $A_JAR $A_INTERFACE_JAR || die "ijar failed"
$IJAR $A_ZIP_JAR $A_ZIP_INTERFACE_JAR || die "ijar failed"
cmp $A_INTERFACE_JAR $A_ZIP_INTERFACE_JAR || \
  die "ijars from jar and zip are different"


# Check that a JAR file can be parsed even if the central directory file count
# is wrong
$IJAR $JAR_WRONG_CENTRAL_DIR $IJAR_WRONG_CENTRAL_DIR || die "ijar failed"
IJAR_FILES=$($UNZIP -qql $IJAR_WRONG_CENTRAL_DIR | wc -l | xargs echo)
if [[ $IJAR_FILES != 2 ]]; then
  die "ijar removed files"
fi

# Check that constant pool references used by JSR308 type annotations are
# preserved
$IJAR $TYPEANN2_JAR $TYPEANN2_IJAR || die "ijar failed"
$JAVAP -classpath $TYPEANN2_IJAR -v Util |
    grep -sq RuntimeVisibleTypeAnnotations ||
    die "RuntimeVisibleTypeAnnotations not preserved!"
set -x
cp $TYPEANN2_JAVA $TEST_TMPDIR/TypeAnnotationTest2.java
$JAVAC -J-Xbootclasspath/p:$LANGTOOLS8 $TEST_TMPDIR/TypeAnnotationTest2.java -cp $TYPEANN2_IJAR ||
  die "javac failed"
set +x

# Check that ijar works on classes with invokedynamic
$IJAR $INVOKEDYNAMIC_JAR $INVOKEDYNAMIC_IJAR || die "ijar failed"
lines=$($JAVAP -c -private -classpath $INVOKEDYNAMIC_JAR ClassWithLambda | grep Code: | wc -l)
check_eq 4 $lines "Input jar should have 4 method bodies!"
lines=$($JAVAP -c -private -classpath $INVOKEDYNAMIC_IJAR ClassWithLambda | grep Code: | wc -l)
check_eq 0 $lines "Interface jar should have no method bodies!"

# Check that Object.class can be processed
$IJAR $OBJECT_JAR $OBJECT_IJAR || die "ijar failed"

# Check that the tool detects and reports a corrupted end of central directory
# record condition
CORRUPTED_JAR=$TEST_TMPDIR/corrupted.jar
# First make the jar one byte longer
cp $JAR_WRONG_CENTRAL_DIR $CORRUPTED_JAR
chmod +w $CORRUPTED_JAR
echo >> $CORRUPTED_JAR
set +e
capture_test_stderr
$IJAR $CORRUPTED_JAR && die "ijar should have failed"
status=$?
set -e
check_ne 0 $status
grep_test_stderr "missing end of central directory record"
restore_outputs
# Then make the jar one byte shorter than the original one
let "NEW_SIZE = `statfmt $CORRUPTED_JAR` - 2"
set_file_length $CORRUPTED_JAR $NEW_SIZE
set +e
capture_test_stderr
$IJAR $CORRUPTED_JAR && die "ijar should have failed"
status=$?
set -e
check_ne 0 $status
grep_test_stderr "missing end of central directory record"
restore_outputs

echo "PASS"
