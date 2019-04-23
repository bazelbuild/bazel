#!/bin/bash
#
# Support functions for building ProGuard.

SRC=src
OUT=out
LIB=../lib

TARGET=1.8

PROGUARD_JAR=$LIB/proguard.jar
RETRACE_JAR=$LIB/retrace.jar
PROGUARD_GUI_JAR=$LIB/proguardgui.jar
ANNOTATIONS_JAR=$LIB/annotations.jar

set -o pipefail

function download {
  if [ ! -f "$2" ]; then
    echo "Downloading $2..."
    if type wget > /dev/null 2>&1; then
      wget -O "$2" "$1"
    else
      curl -L -o "$2" "$1"
    fi
  fi
}

function compile {
  # Compile java source files.
  echo "Compiling $(basename $PWD) ($1)..."
  mkdir -p "$OUT" && \
  javac -nowarn -Xlint:none -source $TARGET -target $TARGET \
    -sourcepath "$SRC" -d "$OUT" \
    ${2:+-classpath "$2"} \
    `find $SRC -name _*.java` \
    "$SRC"/${1//.//}.java 2>&1 \
    | sed -e 's|^|  |' || return 1

  # Copy resource files.
  (cd "$SRC" && \
   find proguard \
     \( -name \*.properties -o -name \*.png -o -name \*.gif -o -name \*.pro \) \
     -exec cp --parents {} "../$OUT" \; )
}

function createjar {
  echo "Creating $1..."
  if [ -f "$SRC/META-INF/MANIFEST.MF" ]; then
    jar -cfm "$1" "$SRC/META-INF/MANIFEST.MF" -C "$OUT" proguard
  else
    jar -cf "$1" -C "$OUT" proguard
  fi
}

function updatejar {
  echo "Updating $1..."
  jar -uf "$1" -C "$OUT" proguard
}
