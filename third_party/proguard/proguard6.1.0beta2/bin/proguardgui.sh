#!/bin/sh
#
# Start-up script for the GUI of ProGuard -- free class file shrinker,
# optimizer, obfuscator, and preverifier for Java bytecode.
#
# Note: when passing file names containing spaces to this script,
#       you'll have to add escaped quotes around them, e.g.
#       "\"/My Directory/My File.txt\""

# Account for possibly missing/basic readlink.
# POSIX conformant (dash/ksh/zsh/bash).
PROGUARD=`readlink -f "$0" 2>/dev/null`
if test "$PROGUARD" = ''
then
  PROGUARD=`readlink "$0" 2>/dev/null`
  if test "$PROGUARD" = ''
  then
    PROGUARD="$0"
  fi
fi

PROGUARD_HOME=`dirname "$PROGUARD"`/..

# On Linux, Java 1.6.0_24 and higher hang when starting the GUI:
#   http://bugs.sun.com/bugdatabase/view_bug.do?bug_id=7027598
# We're using the -D option as a workaround.
java -DsuppressSwingDropSupport=true -jar "$PROGUARD_HOME/lib/proguardgui.jar" "$@"
