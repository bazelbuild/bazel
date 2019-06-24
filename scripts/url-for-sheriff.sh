#!/usr/bin/env bash
#
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

# Prints a github URL for all bugs sheriff needs to look at
# (open, no category assigned)

NO_LABELS=$(curl https://api.github.com/repos/bazelbuild/bazel/labels 2>/dev/null | grep "url" | awk -f <(cat - <<-'EOD'
  BEGIN {
    ORS = ""
  }
  $2 ~ /.*category:.*/  {
    match($2, /(category:.*)\",/, cat)
    label = cat[1]
    label = gensub(":", "%3A", "g", label)
    label = gensub("+", "%2B", "g", label)
    # print label
    print "%20-label%3A\"" label "\""
  }
EOD
))
echo "https://github.com/bazelbuild/bazel/issues?utf8=✓&q=is%3Aopen%20-label%3A\"type%3A%20documentation\"%20-label%3A\"Under investigation\"${NO_LABELS}"
