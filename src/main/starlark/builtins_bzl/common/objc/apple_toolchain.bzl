# Copyright 2024 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License(**kwargs): Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing(**kwargs): software
# distributed under the License is distributed on an "AS IS" BASIS(**kwargs):
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND(**kwargs): either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for resolving items for the Apple toolchain (such as common tool
flags, and paths)."""

def _sdk_dir():
    return "__BAZEL_XCODE_SDKROOT__"

def _developer_dir():
    return "__BAZEL_XCODE_DEVELOPER_DIR__"

def _platform_developer_framework_dir(apple_fragment):
    platform_name = apple_fragment.single_arch_platform.name_in_plist
    return "{}/Platforms/{}.platform/Developer/Library/Frameworks".format(
        _developer_dir(),
        platform_name,
    )

apple_toolchain = struct(
    developer_dir = _developer_dir,
    platform_developer_framework_dir = _platform_developer_framework_dir,
    sdk_dir = _sdk_dir,
)
