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

"""Rules for supporting the Scala language."""

load(
    "@io_bazel_rules_scala//scala:scala.bzl",
    orig_scala_library="scala_library",
    orig_scala_macro_library="scala_macro_library",
    orig_scala_binary="scala_binary",
    orig_scala_test="scala_test",
    orig_scala_repositories="scala_repositories",
)

def warning(rule):
  return """This rule has moved out of @bazel_tools!

The scala rules have moved to https://github.com/bazelbuild/rules_scala, you
should now refer them via @io_bazel_rules_scala, use:

load('@io_bazel_rules_scala//scala:scala.bzl', '%s')

instead of:

load('@bazel_tools//tools/build_defs/scala:scala.bzl', '%s')
""" % (rule, rule)

def scala_library(**kwargs):
  print(warning("scala_library"))
  original_scala_library(**kwargs)

def scala_macro_library(**kwargs):
  print(warning("scala_macro_library"))
  orig_scala_macro_library(**kwargs)

def scala_binary(**kwargs):
  print(warning("scala_binary"))
  original_scala_binary(**kwargs)

def scala_test(**kwargs):
  print(warning("scala_test"))
  original_scala_test(**kwargs)

def scala_repositories():
  print(warning("scala_repositories"))
  original_scala_repositories()
