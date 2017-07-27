#!/bin/bash -eu

# Copyright 2017 The Bazel Authors. All rights reserved.
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

ICON=$1
OUTPUT=$2
VCVARS_SETTER=$3

export TMP="$(mktemp -d)"
OWD=$(pwd)
trap "cd \"$OWD\" && rm -fr \"$TMP\"" EXIT

# Stage the source files in the temp directory.
# The resource compiler generates its outputs next to its sources, and we don't
# want to pollute the source tree with output files.
mkdir -p "$TMP/$(dirname "$ICON")"
cp $ICON $TMP/$ICON

# Create the batch file that sets up the VC environment and runs the Resource
# Compiler.
RUNNER=$TMP/vs.bat
"$VCVARS_SETTER" $RUNNER
echo "@rc /nologo bazel.rc" >> $RUNNER
chmod +x $RUNNER

# Run the script and move the output to its final location.
cd $TMP
cat > bazel.rc <<EOF
1 ICON "${ICON//\//\\}"
EOF

./vs.bat
cd $OWD
mv $TMP/bazel.res $OUTPUT
