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

load("//tools/distributions:system_repo.bzl", "system_repo")

def debian_deps():
    debian_java_deps()
    debian_cc_deps()
    debian_proto_deps()
    debian_bin_deps()

def debian_java_deps():
    # An external repository for providing Debian system installed java libraries.
    native.new_local_repository(
        name = "debian_java_deps",
        path = "/usr/share/java",
        build_file = "//tools/distributions/debian:debian_java.BUILD",
    )

def debian_cc_deps():
    # An external repository for providing Debian system installed C/C++ libraries.
    system_repo(
        name = "debian_cc_deps",
        # /usr/lib is the default library search path for every cc compile in Debian,
        # we use -l as linkopts in debian_cc.BUILD so we actually don't have to link
        # any system paths for this repo.
        symlinks = {},
        build_file = "//tools/distributions/debian:debian_cc.BUILD",
    )

def debian_proto_deps():
    # An external repository for providing Debian system installed proto files.
    system_repo(
        name = "debian_proto_deps",
        symlinks = {
            "google/protobuf": "/usr/include/google/protobuf",
        },
        build_file = "//tools/distributions/debian:debian_proto.BUILD",
    )

def debian_bin_deps():
    # An external repository for providing Debian system installed binaries.
    system_repo(
        name = "debian_bin_deps",
        symlinks = {
            "protoc": "/usr/bin/protoc",
        },
        build_file = "//tools/distributions/debian:debian_bin.BUILD",
    )
