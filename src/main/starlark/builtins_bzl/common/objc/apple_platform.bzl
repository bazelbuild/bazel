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

"""Apple platform definitions."""

# LINT.IfChange
# This struct retains a duplicate list in ApplePlatform.PlatformType during the migration.
# TODO(b/331163027): Remove the IfChange clause once the duplicate is removed.

# Describes an Apple "platform type", such as iOS, macOS, tvOS, visionOS, or watchOS. This
# is distinct from a "platform", which is the platform type combined with one or
# more CPU architectures.
# Specific instances of this type can be retrieved by
# accessing the fields of the apple_common.platform_type:
# apple_common.platform_type.ios
# apple_common.platform_type.macos
# apple_common.platform_type.tvos
# apple_common.platform_type.watchos
# An ApplePlatform is implied from a
# platform type (for example, watchOS) together with a cpu value (for example, armv7).
PLATFORM_TYPE = struct(
    ios = "ios",
    visionos = "visionos",
    watchos = "watchos",
    tvos = "tvos",
    macos = "macos",
    catalyst = "catalyst",
)
# LINT.ThenChange(//src/main/java/com/google/devtools/build/lib/rules/apple/ApplePlatform.java)
