#!/bin/bash
#
# GNU/Linux build script for ReTrace.

cd $(dirname "$0")

source ../buildscripts/functions.sh

MAIN_CLASS=proguard.retrace.ReTrace

# Make sure the ProGuard core has been compiled.
if [ ! -d ../core/$OUT ]; then
  ../core/build.sh || exit 1
fi

# Compile and package.
export CLASSPATH=../core/$OUT

compile   $MAIN_CLASS && \
createjar "$RETRACE_JAR" || exit 1
