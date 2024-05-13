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

"""Module extension to declare Android runtime dependencies for Bazel."""

load("//tools/build_defs/repo:http.bzl", "http_archive", "http_jar")

def _remote_android_tools_extensions_impl(module_ctx):
    http_archive(
        name = "android_tools",
        sha256 = "2b661a761a735b41c41b3a78089f4fc1982626c76ddb944604ae3ff8c545d3c2",  # DO_NOT_REMOVE_THIS_ANDROID_TOOLS_UPDATE_MARKER
        url = "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.30.0.tar",
    )
    http_jar(
        name = "android_gmaven_r8",
        sha256 = "59753e70a74f918389cc87f1b7d66b5c0862932559167425708ded159e3de439",
        url = "https://maven.google.com/com/android/tools/r8/8.3.37/r8-8.3.37.jar",
    )
    return module_ctx.extension_metadata(reproducible = True)

remote_android_tools_extensions = module_extension(
    implementation = _remote_android_tools_extensions_impl,
)
