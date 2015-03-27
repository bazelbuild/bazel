#!/bin/bash
set -euo pipefail

OUTPUT=${PWD}/$1
shift
ZIP=${PWD}/$1
shift
UNZIP=${PWD}/$1
shift
CONSTANTS_JAVA=${PWD}/$1

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

OUTPUT_CONSTANTS=$TMPDIR/java/com/google/devtools/build/lib/Constants.java

mkdir -p $(dirname $OUTPUT_CONSTANTS)
cp $CONSTANTS_JAVA $OUTPUT_CONSTANTS

cd $TMPDIR
$ZIP -jt -qr $OUTPUT .
