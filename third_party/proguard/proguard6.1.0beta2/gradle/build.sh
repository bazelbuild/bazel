#!/bin/bash
#
# GNU/Linux build script for the ProGuard Gradle task.

cd $(dirname "$0")

source ../buildscripts/functions.sh

MAIN_CLASS=proguard.gradle.ProGuardTask

GRADLE_HOME=${GRADLE_HOME:-/usr/local/java/gradle}

GRADLE_PATH=\
$(echo $GRADLE_HOME/lib/plugins/gradle-plugins-*.jar):\
$(echo $GRADLE_HOME/lib/gradle-logging-*.jar):\
$(echo $GRADLE_HOME/lib/gradle-base-services-?.*.jar):\
$(echo $GRADLE_HOME/lib/gradle-base-services-groovy-*.jar):\
$(echo $GRADLE_HOME/lib/gradle-core-[0-9]*.jar):\
$(echo $GRADLE_HOME/lib/gradle-core-api-*.jar):\
$(echo $GRADLE_HOME/lib/groovy-all-*.jar):\
$(echo $GRADLE_HOME/lib/slf4j-api-*.jar)

# Make sure the Gradle jars are present.
if [ ! -f "${GRADLE_PATH%%:*}" ]; then
  echo "Please make sure the environment variable GRADLE_HOME is set correctly,"
  echo "if you want to compile the optional ProGuard Gradle task."
  exit 1
fi

# Make sure the ProGuard core has been compiled.
if [ ! -d ../core/$OUT ]; then
  ../core/build.sh || exit 1
fi

# Compile and package.
export CLASSPATH=../core/$OUT:$GRADLE_PATH

compile   $MAIN_CLASS && \
updatejar "$PROGUARD_JAR" || exit 1
