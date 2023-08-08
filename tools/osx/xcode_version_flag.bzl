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

"""Rules that allows select() to differentiate between Apple OS versions."""

def _strip_version(version):
    """Strip trailing characters that aren't digits or '.' from version names.

    Some OS versions look like "9.0gm", which is not useful for select()
    statements. Thus, we strip the trailing "gm" part.

    Args:
      version: the version string

    Returns:
      The version with trailing letters stripped.
    """
    result = ""
    string = str(version)
    for i in range(len(string)):
        ch = string[i]
        if not ch.isdigit() and ch != ".":
            break

        result += ch

    return result

def _strip_or_pad_version(version, num_components):
    """Strips or pads a version string to the given number of components.

    If the version string contains fewer than the requested number of
    components, it will be padded with zeros.

    Args:
        version: The version string.
        num_components: The desired number of components.

    Returns:
        The version, stripped or padded to the requested number of components.
    """
    version_string = str(version)
    components = version_string.split(".")
    if num_components <= len(components):
        return ".".join(components[:num_components])
    return version_string + (".0" * (num_components - len(components)))

_VERSION_PRECISION_COMPONENTS = {
    "major": 1,
    "minor": 2,
    "patch": 3,
}

def _xcode_version_flag_impl(ctx):
    """A rule that allows select() to differentiate between Xcode versions."""
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]
    xcode_version = xcode_config.xcode_version()
    precision = ctx.attr.precision

    if not precision:
        value = _strip_version(xcode_version)
    elif precision == "exact":
        value = str(xcode_version)
    else:
        num_components = _VERSION_PRECISION_COMPONENTS[precision]
        value = _strip_or_pad_version(xcode_version, num_components)

    return config_common.FeatureFlagInfo(value = value)

def _ios_sdk_version_flag_impl(ctx):
    """A rule that allows select() to select based on the iOS SDK version."""
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    return config_common.FeatureFlagInfo(value = _strip_version(
        xcode_config.sdk_version_for_platform(
            apple_common.platform.ios_device,
        ),
    ))

def _tvos_sdk_version_flag_impl(ctx):
    """A rule that allows select() to select based on the tvOS SDK version."""
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    return config_common.FeatureFlagInfo(value = _strip_version(
        xcode_config.sdk_version_for_platform(
            apple_common.platform.tvos_device,
        ),
    ))

def _visionos_sdk_version_flag_impl(ctx):
    """A rule that allows select() to select based on the visionOS SDK version."""
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    return config_common.FeatureFlagInfo(value = _strip_version(
        xcode_config.sdk_version_for_platform(
            apple_common.platform.visionos_device,
        ),
    ))

def _watchos_sdk_version_flag_impl(ctx):
    """A rule that allows select() to select based on the watchOS SDK version."""
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    return config_common.FeatureFlagInfo(value = _strip_version(
        xcode_config.sdk_version_for_platform(
            apple_common.platform.watchos_device,
        ),
    ))

def _macos_sdk_version_flag_impl(ctx):
    """A rule that allows select() to select based on the macOS SDK version."""
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]

    return config_common.FeatureFlagInfo(value = _strip_version(
        xcode_config.sdk_version_for_platform(
            apple_common.platform.macos,
        ),
    ))

xcode_version_flag = rule(
    implementation = _xcode_version_flag_impl,
    attrs = {
        "precision": attr.string(
            doc = """\
The desired precision with which the version number will be provided. The
permitted values of this attribute are given below, with examples of the value
that the rule would provide if the selected `xcode_version` target reported
`version = "1.2.3.4X789"`:

-   `major`: Provide only the major component of the version number (e.g., `1`).
-   `minor`: Provide only the major and minor components of the version number
    (e.g., `1.2`).
-   `patch`: Provide only the major, minor, and patch components of the version
    number (e.g., `1.2.3`).
-   `exact`: Provide the version number reported by the `xcode_version` target
    unmodified (e.g., `1.2.3.4X789`).

If `minor` or `patch` precision is requested and the version number reported by
the `xcode_version` has a lower precision, the result will be padded with `.0`
segments as necessary to ensure that it is consistent. For example, if the
selected `xcode_version` reported `version = "1"`, then `minor` precision would
be `1.0` and `patch` precision would be `1.0.0`.

Setting the precision lets you write `config_setting`s that test for more or
less specific versions of Xcode. For example, if you want to handle all `11.*`
versions of Xcode identically but do something different for Xcode 12.0 and
12.1, you can express that as follows:

```
config_setting(
    name = "xcode_11_any",
    flag_values = {
        "//tools/osx:xcode_version_flag_major": "11",
    },
)

config_setting(
    name = "xcode_12_0",
    flag_values = {
        "//tools/osx:xcode_version_flag_minor": "12.0",
    },
)

config_setting(
    name = "xcode_12_1",
    flag_values = {
        "//tools/osx:xcode_version_flag_minor": "12.1",
    },
)
```
""",
            mandatory = False,
            values = ["major", "minor", "patch", "exact"],
        ),
        "_xcode_config": attr.label(default = configuration_field(
            fragment = "apple",
            name = "xcode_config_label",
        )),
    },
)

ios_sdk_version_flag = rule(
    implementation = _ios_sdk_version_flag_impl,
    attrs = {
        "_xcode_config": attr.label(default = configuration_field(
            fragment = "apple",
            name = "xcode_config_label",
        )),
    },
)

tvos_sdk_version_flag = rule(
    implementation = _tvos_sdk_version_flag_impl,
    attrs = {
        "_xcode_config": attr.label(default = configuration_field(
            fragment = "apple",
            name = "xcode_config_label",
        )),
    },
)

visionos_sdk_version_flag = rule(
    implementation = _visionos_sdk_version_flag_impl,
    attrs = {
        "_xcode_config": attr.label(default = configuration_field(
            fragment = "apple",
            name = "xcode_config_label",
        )),
    },
)

watchos_sdk_version_flag = rule(
    implementation = _watchos_sdk_version_flag_impl,
    attrs = {
        "_xcode_config": attr.label(default = configuration_field(
            fragment = "apple",
            name = "xcode_config_label",
        )),
    },
)

macos_sdk_version_flag = rule(
    implementation = _macos_sdk_version_flag_impl,
    attrs = {
        "_xcode_config": attr.label(default = configuration_field(
            fragment = "apple",
            name = "xcode_config_label",
        )),
    },
)
