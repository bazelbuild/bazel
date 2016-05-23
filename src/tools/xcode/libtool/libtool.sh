#!/bin/bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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
#
# libtool.sh runs the command passed to it using "xcrunwrapper libtool".
#
# It creates symbolic links for all input files with a content-hash appended
# to their original name (foo.o becomes foo_{md5sum}.o). This is to circumvent
# a bug in the original tool that arises when two input files have the same
# base name (even if they are in different directories).

set -eu

MY_LOCATION=${MY_LOCATION:-"$0.runfiles/bazel_tools/tools/objc"}
WRAPPER="${MY_LOCATION}/xcrunwrapper.sh"

# TODO(b/28347228): When all callers of "xcrunwrapper libtool" are migrated to
# using this script, move the symlinking behavior to this script.
"${WRAPPER}" libtool "$@"
