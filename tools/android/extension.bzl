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
"""A module extension to bring in the android tools under
@android_tools."""

load("//tools/build_defs/repo:http.bzl", "http_archive")

def _android_tools_extension_impl(ctx):
	# This must be kept in sync with the top-level WORKSPACE file.
	http_archive(
	    name = "android_tools",
	    sha256 = "1afa4b7e13c82523c8b69e87f8d598c891ec7e2baa41d9e24e08becd723edb4d",  # DO_NOT_REMOVE_THIS_ANDROID_TOOLS_UPDATE_MARKER
	    url = "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.27.0.tar.gz",
	)

	# This must be kept in sync with the top-level WORKSPACE file.
	http_jar(
	    name = "android_gmaven_r8",
	    sha256 = "8626ca32fb47aba7fddd2c897615e2e8ffcdb4d4b213572a2aefb3f838f01972",
	    url = "https://maven.google.com/com/android/tools/r8/3.3.28/r8-3.3.28.jar",
	)

android_tools_extension = module_extension(
    implementation = _android_tools_extension_impl,
)
