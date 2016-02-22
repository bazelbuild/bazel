#!/bin/bash

# Copyright 2015 The Bazel Authors. All rights reserved.
#
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
# limitations under the License.

# Jenkins is capable of executing shell scripts directly, even on Windows,
# but it uses a shell binary bundled with it and not the msys one. We don't
# want to use two different shells, so a batch file is used instead to call
# the msys shell.

# We need to execute bash with -l so that we don't get the usual environment
# variables from cmd.exe which would interfere with our operation, but that
# means that PWD will be $HOME. Thus, we need to cd to the source tree.
cd $(dirname $0)/../../..

# Find Java. Minor versions and thus the name of the directory changes quite
# often.
export JAVA_HOME=$(ls -d /c/Program\ Files/Java/jdk* 2> /dev/null | head -n 1)
if [[ "$JAVA_HOME" == "" ]]; then
  echo "JDK not found under c:\\Program Files\\Java" 1>& 2
  exit 1
fi

# These variables are temporarily needed for Bazel
export BAZEL_SH="c:/tools/msys64/usr/bin/bash.exe"
export TMPDIR="c:/temp"

# Even though there are no quotes around $* in the .bat file, arguments
# containing spaces seem to be passed properly.
exec ./compile.sh "$*"
