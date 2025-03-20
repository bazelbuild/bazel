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

# Prefix for simulator environment cpu values
SIMULATOR_ENVIRONMENT_CPU_PREFIX = "sim_"

# Prefix for device environment cpu values
DEVICE_ENVIRONMENT_CPU_PREFIX = "device_"

def _get_unprefixed_apple_cpu(platform_type, apple_cpus):
    cpu = _get_prefixed_apple_cpu(platform_type, apple_cpus)
    if cpu.startswith(SIMULATOR_ENVIRONMENT_CPU_PREFIX):
        cpu = cpu.substring(SIMULATOR_ENVIRONMENT_CPU_PREFIX.length())
    elif cpu.startswith(DEVICE_ENVIRONMENT_CPU_PREFIX):
        cpu = cpu.substring(DEVICE_ENVIRONMENT_CPU_PREFIX.length())
    return cpu

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
    if platform_type == PLATFORM_TYPE.catalyst:
        return apple_cpus.catalyst_cpus[0]
    fail("Unsupported platform type %s" % platform_type)

def _get_single_arch_platform(apple_config):
    """
    Gets the single "effective" platform for this configuration's PlatformType and architecture.

    Prefer this over getMultiArchPlatform(PlatformType) only in cases if in
    the context of rule logic which is only concerned with a single architecture (such as in
    objc_library, which registers single-architecture compile actions).

    Args:
      apple_config: an AppleConfigurationApi implementation

    Returns:
      an apple_platform.PLATFORM struct
    """
    prefixed_cpu = _get_prefixed_apple_cpu(apple_config.apple_platform_type, apple_config.apple_cpus)
    return apple_platform.for_target(apple_config.apple_platform_type, prefixed_cpu)

def _get_single_architecture(apple_config):
    # TODO(b/331163027): Update comment once getMultiArchitectures is ported to Starlark.
    """
    Gets the single "effective" architecture for this configuration.

    Prefer this over getMultiArchitectures(PlatformType) (not yet ported to Starlark) only
    if in the context of rule logic which is only concerned with a single architecture (such as in
    <code>objc_library</code>, which registers single-architecture compile actions).

    Single effective architecture is determined using the following rules:

    * If --apple_split_cpu is set (done via prior configuration transition), then that
      is the effective architecture.
    * If the multi cpus flag (e.g. --ios_multi_cpus) is set and non-empty, then the
      first such architecture is returned.
    * In the case of iOS, use --cpu if it leads with "ios_" for backwards
      compatibility.
    * In the case of macOS, use --cpu if it leads with "darwin_" for backwards
      compatibility.
    * Use the default.

    Args:
      apple_config: an AppleConfigurationApi implementation

    Returns:
      a string, e.g. "i386" or "arm64"
    """
    return _get_unprefixed_apple_cpu(apple_config.apple_platform_type, apple_config.apple_cpus)

# LINT.ThenChange(//src/main/java/com/google/devtools/build/lib/rules/apple/AppleConfiguration.java)

apple_configuration = struct(
    get_single_architecture = _get_single_architecture,
    get_single_arch_platform = _get_single_arch_platform,
)
