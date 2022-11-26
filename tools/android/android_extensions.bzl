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

def _remote_android_tools_extensions_impl(_ctx):
    http_archive(
        name = "android_tools",
        sha256 = "1afa4b7e13c82523c8b69e87f8d598c891ec7e2baa41d9e24e08becd723edb4d",  # do_not_remove_this_android_tools_update_marker
        url = "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.27.0.tar.gz",
    )
    http_jar(
        name = "android_gmaven_r8",
        sha256 = "8626ca32fb47aba7fddd2c897615e2e8ffcdb4d4b213572a2aefb3f838f01972",
        url = "https://maven.google.com/com/android/tools/r8/3.3.28/r8-3.3.28.jar",
    )

remote_android_tools_extensions = module_extension(
    implementation = _remote_android_tools_extensions_impl,
)
