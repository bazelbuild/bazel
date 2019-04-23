#!/bin/bash
#
# GNU/Linux build script for the ProGuard GUI.

cd $(dirname "$0")

source ../buildscripts/functions.sh

MAIN_CLASS=proguard.gui.ProGuardGUI

# Make sure the ProGuard core has been compiled.
if [ ! -d ../core/$OUT ]; then
  ../core/build.sh || exit 1
fi

# Make sure ReTrace has been compiled.
if [ ! -d ../retrace/$OUT ]; then
  ../retrace/build.sh || exit 1
fi

# Compile and package.
export CLASSPATH=../core/$OUT:../retrace/$OUT

compile   $MAIN_CLASS && \
createjar "$PROGUARD_GUI_JAR" || exit 1
