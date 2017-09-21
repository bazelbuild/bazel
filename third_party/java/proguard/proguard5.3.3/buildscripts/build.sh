#!/bin/bash
#
# GNU/Linux build script for ProGuard.

#
# Configuration.
#

ANT_HOME=${ANT_HOME:-/usr/local/java/ant}
GRADLE_HOME=${GRADLE_HOME:-/usr/local/java/gradle}
WTK_HOME=${WTK_HOME:-/usr/local/java/wtk}

if [ -z $PROGUARD_HOME ]; then
  PROGUARD_HOME=$(which "$0")
  PROGUARD_HOME=$(dirname "$0")/..
fi

cd "$PROGUARD_HOME"

SRC=src
CLASSES=classes
LIB=lib

PROGUARD=proguard/ProGuard
PROGUARD_GUI=proguard/gui/ProGuardGUI
RETRACE=proguard/retrace/ReTrace
ANT_TASK=proguard/ant/ProGuardTask
GRADLE_TASK=proguard/gradle/ProGuardTask
WTK_PLUGIN=proguard/wtk/ProGuardObfuscator

ANT_JAR=$ANT_HOME/lib/ant.jar
GRADLE_PATH=\
$GRADLE_HOME/lib/plugins/gradle-plugins-*.jar:\
$GRADLE_HOME/lib/gradle-logging-*.jar:\
$GRADLE_HOME/lib/gradle-base-services-*.jar:\
$GRADLE_HOME/lib/gradle-base-services-groovy-*.jar:\
$GRADLE_HOME/lib/gradle-core-*.jar:\
$GRADLE_HOME/lib/groovy-all-*.jar
WTK_JAR=$WTK_HOME/wtklib/kenv.zip

PROGUARD_JAR=$LIB/proguard.jar
PROGUARD_GUI_JAR=$LIB/proguardgui.jar
RETRACE_JAR=$LIB/retrace.jar

#
# Function definitions.
#

function compile {
  # Compile java source files.
  echo "Compiling ${1//\//.} ..."
  javac -nowarn -Xlint:none -sourcepath "$SRC" -d "$CLASSES" \
    "$SRC/$1.java" 2>&1 \
  | sed -e 's|^|  |'

  # Copy resource files.
  (cd "$SRC"; find $(dirname $1) -maxdepth 1 \
     \( -name \*.properties -o -name \*.png -o -name \*.gif -o -name \*.pro \) \
     -exec cp --parents {} "../$CLASSES" \; )
}

function createjar {
  echo "Creating $2..."
  jar -cfm "$2" "$SRC/$(dirname $1)/MANIFEST.MF" -C "$CLASSES" $(dirname $1)
}

function updatejar {
  echo "Updating $PROGUARD_JAR..."
  jar -uf "$PROGUARD_JAR" -C "$CLASSES" $(dirname $1)
}

#
# Main script body.
#

mkdir -p "$CLASSES"

compile   $PROGUARD
createjar $PROGUARD "$PROGUARD_JAR"

compile   $PROGUARD_GUI
createjar $PROGUARD_GUI "$PROGUARD_GUI_JAR"

compile   $RETRACE
createjar $RETRACE "$RETRACE_JAR"

if [ -f "$ANT_JAR" ]; then
  export CLASSPATH=$ANT_JAR
  compile   $ANT_TASK
  updatejar $ANT_TASK
else
  echo "Please make sure the environment variable ANT_HOME is set correctly,"
  echo "if you want to compile the optional ProGuard Ant task."
fi

if [ -f "${GRADLE_PATH%%:*}" ]; then
  export CLASSPATH=$GRADLE_PATH
  compile   $GRADLE_TASK
  updatejar $GRADLE_TASK
else
  echo "Please make sure the environment variable GRADLE_HOME is set correctly,"
  echo "if you want to compile the optional ProGuard Gradle task."
fi

if [ -f "$WTK_JAR" ]; then
  export CLASSPATH=$WTK_JAR
  compile   $WTK_PLUGIN
  updatejar $WTK_PLUGIN
else
  echo "Please make sure the environment variable WTK_HOME is set correctly,"
  echo "if you want to compile the optional ProGuard WTK plugin."
fi
