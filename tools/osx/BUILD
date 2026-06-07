load("@bazel_lib//lib:copy_file.bzl", "copy_file")
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

copy_file(
    name = "xcode-locator-genrule",
    src = "xcode_locator_stub.sh",
    out = "xcode-locator",
    is_executable = True,
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
