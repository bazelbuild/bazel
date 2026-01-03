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

"""WIP: A configuration containing flags required for Apple platforms and tools.

Initially some code ported from .../devtools/build/lib/rules/apple/AppleConfiguration.java
"""
# LINT.IfChange

load(":common/objc/apple_platform.bzl", "PLATFORM_TYPE", "apple_platform")

def _get_prefixed_apple_cpu(platform_type, apple_cpus):
    if apple_cpus.apple_split_cpu:
        return apple_cpus.apple_split_cpu
    if platform_type == PLATFORM_TYPE.ios:
        return apple_cpus.ios_multi_cpus[0]
    if platform_type == PLATFORM_TYPE.visionos:
        return apple_cpus.visionos_cpus[0]
    if platform_type == PLATFORM_TYPE.watchos:
        return apple_cpus.watchos_cpus[0]
    if platform_type == PLATFORM_TYPE.tvos:
        return apple_cpus.tvos_cpus[0]
    if platform_type == PLATFORM_TYPE.macos:
        return apple_cpus.macos_cpus[0]
    fail("Unsupported platform type %s" % platform_type)

def _get_single_arch_platform(apple_config):
    """
    Gets the single "effective" platform for this configuration's PlatformType and architecture.

    Args:
      apple_config: an AppleConfigurationApi implementation

    Returns:
      an apple_platform.PLATFORM struct
    """
    prefixed_cpu = _get_prefixed_apple_cpu(apple_config.apple_platform_type, apple_config.apple_cpus)
    return apple_platform.for_target(apple_config.apple_platform_type, prefixed_cpu)

# LINT.ThenChange(//src/main/java/com/google/devtools/build/lib/rules/apple/AppleConfiguration.java)

apple_configuration = struct(
    get_single_arch_platform = _get_single_arch_platform,
)
