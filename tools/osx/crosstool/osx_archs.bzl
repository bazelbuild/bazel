"""Information regarding crosstool-supported architectures."""
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

# List of architectures supported by osx crosstool.
OSX_TOOLS_NON_DEVICE_ARCHS = [
    "darwin_x86_64",
    "ios_i386",
    "ios_x86_64",
    "watchos_i386",
    "watchos_x86_64",
    "tvos_x86_64",
]

OSX_TOOLS_ARCHS = [
    "armeabi-v7a",
    "ios_armv7",
    "ios_arm64",
    "ios_arm64e",
    "watchos_armv7k",
    "watchos_arm64_32",
    "tvos_arm64",
] + OSX_TOOLS_NON_DEVICE_ARCHS

# Target constraints for each arch.
# TODO(apple-rules): Rename osx constraint to macOS.
# TODO(apple-rules): Add constraints for watchos and tvos.
OSX_TOOLS_CONSTRAINTS = {
    "darwin_x86_64": ["@platforms//os:osx", "@platforms//cpu:x86_64"],
    "ios_i386": ["@platforms//os:ios", "@platforms//cpu:x86_32"],
    "ios_x86_64": ["@platforms//os:ios", "@platforms//cpu:x86_64"],
    "watchos_i386": ["@platforms//os:ios", "@platforms//cpu:x86_32"],
    "watchos_x86_64": ["@platforms//os:ios", "@platforms//cpu:x86_64"],
    "tvos_x86_64": ["@platforms//os:ios", "@platforms//cpu:x86_64"],
    "armeabi-v7a": ["@platforms//cpu:arm"],
    "ios_armv7": ["@platforms//os:ios", "@platforms//cpu:arm"],
    "ios_arm64": ["@platforms//os:ios", "@platforms//cpu:aarch64"],
    "ios_arm64e": ["@platforms//os:ios", "@platforms//cpu:aarch64"],
    "watchos_armv7k": ["@platforms//os:ios", "@platforms//cpu:arm"],
    "watchos_arm64_32": ["@platforms//os:ios", "@platforms//cpu:aarch64"],
    "tvos_arm64": ["@platforms//os:ios", "@platforms//cpu:aarch64"],
}
