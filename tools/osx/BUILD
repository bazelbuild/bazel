package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"] + ["**/*"]),
)

filegroup(
    name = "bzl_srcs",
    srcs = glob(["*.bzl"]) + [
        "//tools/osx/crosstool:bzl_srcs",
    ],
    visibility = ["//tools:__pkg__"],
)

exports_files([
    "xcode_locator.m",
])

DARWIN_XCODE_LOCATOR_COMPILE_COMMAND = """
  /usr/bin/xcrun --sdk macosx clang -mmacosx-version-min=10.9 -fobjc-arc -framework CoreServices \
      -framework Foundation -o $@ $<
"""

genrule(
    name = "xcode-locator-genrule",
    srcs = select({
        ":darwin": ["xcode_locator.m"],
        ":darwin_x86_64": ["xcode_locator.m"],
        "//conditions:default": ["xcode_locator_stub.sh"],
    }),
    outs = ["xcode-locator"],
    cmd = select({
        ":darwin": DARWIN_XCODE_LOCATOR_COMPILE_COMMAND,
        ":darwin_x86_64": DARWIN_XCODE_LOCATOR_COMPILE_COMMAND,
        "//conditions:default": "cp $< $@",
    }),
    local = 1,
    output_to_bindir = 1,
)

# TODO(cparsons): Consolidate with config_settings under //src
config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin_x86_64",
    values = {"cpu": "darwin_x86_64"},
    visibility = ["//visibility:public"],
)

xcode_config_alias(name = "current_xcode_config")

load(
    "//tools/osx:xcode_version_flag.bzl",
    "ios_sdk_version_flag",
    "macos_sdk_version_flag",
    "tvos_sdk_version_flag",
    "watchos_sdk_version_flag",
    "xcode_version_flag",
)

xcode_version_flag(name = "xcode_version_flag")

ios_sdk_version_flag(name = "ios_sdk_version_flag")

tvos_sdk_version_flag(name = "tvos_sdk_version_flag")

watchos_sdk_version_flag(name = "watchos_sdk_version_flag")

macos_sdk_version_flag(name = "macos_sdk_version_flag")
