# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""Defines a java_binary rule class that is executable but doesn't have launcher
 flag resolution.

There are three physical rule classes for java_binary and we want all of them
to have a name string of "java_binary" because various tooling expects that.
But we also need the rule classes to be defined in separate files. That way the
hash of their bzl environments will be different. See http://b/226379109,
specifically #20, for details.
"""

load(":bazel/java/bazel_java_binary.bzl", "make_java_binary", "make_java_test")

java_binary = make_java_binary(executable = True, resolve_launcher_flag = False)

java_test = make_java_test(resolve_launcher_flag = False)
