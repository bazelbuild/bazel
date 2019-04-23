#!/bin/bash
#
# GNU/Linux build script for ProGuard.

cd $(dirname "$0")

source ../buildscripts/functions.sh

MAIN_CLASS=proguard.ProGuard

GSON_VERSION=2.8.5
GSON_URL=https://jcenter.bintray.com/com/google/code/gson/gson/${GSON_VERSION}/gson-${GSON_VERSION}.jar
GSON_JAR=$LIB/gson-${GSON_VERSION}.jar

download  "$GSON_URL" "$GSON_JAR" && \
compile   $MAIN_CLASS "$GSON_JAR" && \
createjar "$PROGUARD_JAR" || exit 1
