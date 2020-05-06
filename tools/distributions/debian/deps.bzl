# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""Macros for defining external repositories for Debian system installed libraries."""

def debian_deps():
    debian_java_deps()
    debian_cc_deps()

def debian_java_deps():
    # An external repository for providing Debian system installed java libraries.
    native.new_local_repository(
        name = "debian_java_deps",
        path = "/usr/share/java",
        build_file = "//tools/distributions/debian:debian_java.BUILD",
    )

def debian_cc_deps():
    # An external repository for providing Debian system installed java libraries.
    native.new_local_repository(
        name = "debian_cc_deps",
        path = "/usr/lib",
        build_file = "//tools/distributions/debian:debian_cc.BUILD",
    )
