#!/bin/bash
#
# GNU/Linux build script for the ProGuard Ant task.

cd $(dirname "$0")

source ../buildscripts/functions.sh

MAIN_CLASS=proguard.ant.ProGuardTask

ANT_HOME=${ANT_HOME:-/usr/local/java/ant}

ANT_JAR=$ANT_HOME/lib/ant.jar

# Make sure the Ant jar is present.
if [ ! -f "$ANT_JAR" ]; then
  echo "Please make sure the environment variable ANT_HOME is set correctly,"
  echo "if you want to compile the optional ProGuard Ant task."
  exit 1
fi

# Make sure the ProGuard core has been compiled.
if [ ! -d ../core/$OUT ]; then
  ../core/build.sh || exit 1
fi

# Compile and package.
export CLASSPATH=../core/$OUT:$ANT_JAR

compile   $MAIN_CLASS && \
updatejar "$PROGUARD_JAR" || exit 1
