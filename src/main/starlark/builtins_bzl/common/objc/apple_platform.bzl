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

# PLATFORM corresponds to Xcode's notion of a platform as would be found in
# Xcode.app/Contents/Developer/Platforms</code>. Each platform represents an
# Apple platform type (such as iOS or tvOS) combined with one or more related CPU
# architectures. For example, the iOS simulator platform supports x86_64
# and i386 architectures.
# More commonly, however, the apple configuration fragment has fields/methods that allow rules
# to determine the platform for which a target is being built.
def _create_platform(name, name_in_plist, platform_type, is_device):
    return struct(
        name = name,
        name_in_plist = name_in_plist,
        platform_type = platform_type,
        is_device = is_device,
    )

PLATFORM = struct(
    ios_device = _create_platform("ios_device", "iPhoneOS", PLATFORM_TYPE.ios, True),
    ios_simulator = _create_platform("ios_simulator", "iPhoneSimulator", PLATFORM_TYPE.ios, False),
    macos = _create_platform("macos", "MacOSX", PLATFORM_TYPE.macos, True),
    tvos_device = _create_platform("tvos_device", "AppleTVOS", PLATFORM_TYPE.tvos, True),
    tvos_simulator = _create_platform("tvos_simulator", "AppleTVSimulator", PLATFORM_TYPE.tvos, False),
    visionos_device = _create_platform("visionos_device", "XROS", PLATFORM_TYPE.visionos, True),
    visionos_simulator = _create_platform("visionos_simulator", "XRSimulator", PLATFORM_TYPE.visionos, False),
    watchos_device = _create_platform("watchos_device", "WatchOS", PLATFORM_TYPE.watchos, True),
    watchos_simulator = _create_platform("watchos_simulator", "WatchSimulator", PLATFORM_TYPE.watchos, False),
    catalyst = _create_platform("catalyst", "MacOSX", PLATFORM_TYPE.catalyst, True),
)

_TARGET_CPUS_BY_PLATFORM = {
    "ios_simulator": {
        "ios_x86_64": True,
        "ios_i386": True,
        "ios_sim_arm64": True,
    },
    "ios_device": {
        "ios_armv6": True,
        "ios_arm64": True,
        "ios_armv7": True,
        "ios_armv7s": True,
        "ios_arm64e": True,
    },
    "visionos_simulator": {
        "visionos_sim_arm64": True,
    },
    "visionos_device": {
        "visionos_arm64": True,
    },
    "watchos_simulator": {
        "watchos_i386": True,
        "watchos_x86_64": True,
        "watchos_arm64": True,
    },
    "watchos_device": {
        "watchos_armv7k": True,
        "watchos_arm64_32": True,
        "watchos_device_arm64": True,
        "watchos_device_arm64e": True,
    },
    "tvos_simulator": {
        "tvos_x86_64": True,
        "tvos_sim_arm64": True,
    },
    "tvos_device": {
        "tvos_arm64": True,
    },
    "catalyst": {
        "catalyst_x86_64": True,
    },
    "macos": {
        "darwin_x86_64": True,
        "darwin_arm64": True,
        "darwin_arm64e": True,
    },
}

def _for_target_cpu(target_cpu):
    """Returns the platform for the given target CPU.

    Args:
      target_cpu: The target CPU.

    Returns:
      The platform for the given target CPU.
    """
    for platform, target_cpus in _TARGET_CPUS_BY_PLATFORM.items():
        if target_cpu in target_cpus:
            return getattr(PLATFORM, platform)
    fail("No platform found for target CPU %s" % target_cpu)

def _get_target_platform(platform):
    """Returns the target platform as it would be represented in a target triple.

    Note that the target platform for Catalyst is "ios", despite it being represented here as
    its own value.
    """
    if platform.platform_type == PLATFORM_TYPE.catalyst:
        return PLATFORM_TYPE.ios
    return platform.platform_type

def _get_target_environment(platform):
    """Returns the platform's target environment as it would be represented in a target triple.

    Note that the target environment corresponds to the target platform (as returned by
    _get_target_platform(), so "macabi" is an environment of iOS, not a separate platform as it is
    represented in this enumerated type.
    """
    if platform.platform_type == PLATFORM_TYPE.catalyst:
        return "macabi"
    if platform.is_device:
        return "device"
    return "simulator"

# LINT.ThenChange(//src/main/java/com/google/devtools/build/lib/rules/apple/ApplePlatform.java)

apple_platform = struct(
    for_target_cpu = _for_target_cpu,
    get_target_platform = _get_target_platform,
    get_target_environment = _get_target_environment,
)
