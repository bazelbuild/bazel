load(
    "//tools/osx:xcode_version_flag.bzl",
    "ios_sdk_version_flag",
    "macos_sdk_version_flag",
    "tvos_sdk_version_flag",
    "visionos_sdk_version_flag",
    "watchos_sdk_version_flag",
    "xcode_version_flag",
)

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"] + ["**/*"]),
)

filegroup(
    name = "bzl_srcs",
    srcs = glob(["*.bzl"]),
    visibility = ["//tools:__pkg__"],
)

exports_files([
    "xcode_locator.m",
    "xcode_configure.bzl",
])

DARWIN_XCODE_LOCATOR_COMPILE_COMMAND = """
  /usr/bin/xcrun --sdk macosx clang -mmacosx-version-min=10.13 -fobjc-arc -framework CoreServices \
      -framework Foundation -arch arm64 -arch x86_64 -Wl,-no_adhoc_codesign -Wl,-no_uuid -o $@ $< && \
  env -i codesign --identifier $@ --force --sign - $@
"""

genrule(
    name = "xcode-locator-genrule",
    srcs = select({
        "//src/conditions:darwin": ["xcode_locator.m"],
        "//conditions:default": ["xcode_locator_stub.sh"],
    }),
    outs = ["xcode-locator"],
    cmd = select({
        "//src/conditions:darwin": DARWIN_XCODE_LOCATOR_COMPILE_COMMAND,
        "//conditions:default": "cp $< $@",
    }),
    local = 1,
    output_to_bindir = 1,
)

# TODO(cparsons): Consolidate with config_settings under //src
config_setting(
    name = "darwin_x86_64",
    values = {"cpu": "darwin_x86_64"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin_arm64",
    values = {"cpu": "darwin_arm64"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin_arm64e",
    values = {"cpu": "darwin_arm64e"},
    visibility = ["//visibility:public"],
)

xcode_config_alias(name = "current_xcode_config")

xcode_version_flag(name = "xcode_version_flag")

[
    xcode_version_flag(
        name = "xcode_version_flag_{}".format(precision),
        precision = precision,
    )
    for precision in [
        "major",
        "minor",
        "patch",
        "exact",
    ]
]

ios_sdk_version_flag(name = "ios_sdk_version_flag")

tvos_sdk_version_flag(name = "tvos_sdk_version_flag")

visionos_sdk_version_flag(name = "visionos_sdk_version_flag")

watchos_sdk_version_flag(name = "watchos_sdk_version_flag")

macos_sdk_version_flag(name = "macos_sdk_version_flag")
