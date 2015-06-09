#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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

# shift stderr to stdout.
exec 2>&1

# Executing the test log will page it.
echo 'exec ${PAGER:-/usr/bin/less} "$0" || exit 1'

DIR="$TEST_SRCDIR"

# normal commands are run in the exec-root where they have access to
# the entire source tree. By chdir'ing to the runfiles root, tests only
# have direct access to their declared dependencies.
cd "$DIR" || { echo "Could not chdir $DIR"; exit 1; }

# This header marks where --test_output=streamed will start being printed.
echo "-----------------------------------------------------------------------------"

# The path of this command-line is usually relative to the exec-root,
# but when using --run_under it can be a "/bin/bash -c" command-line.

# If the test is at the top of the tree, we have to add '.' to $PATH,
PATH=".:$PATH"

"$@"
