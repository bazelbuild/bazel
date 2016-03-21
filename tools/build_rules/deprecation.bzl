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

def deprecated(language, rule, old_path, new_path):
  return """This rule has moved out of @bazel_tools!

The {0} rules have moved to https://github.com/bazelbuild/rules_{0}, you
should now refer them via @io_bazel_rules_{0}, use:

load('{3}', '{1}')

instead of:

load('{2}', '{1}')
""".format(language, rule, old_path, new_path)
