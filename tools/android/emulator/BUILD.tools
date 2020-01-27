package(default_visibility = ["//visibility:public"])

exports_files(["no_se_linux.properties"])

alias(
    name = "emulator_arm",
    actual = "@androidsdk//:emulator_arm",
)

alias(
    name = "emulator_x86",
    actual = "@androidsdk//:emulator_x86",
)

alias(
    name = "emulator_x86_bios",
    actual = "@androidsdk//:emulator_x86_bios",
)

alias(
    name = "mksd",
    actual = "@androidsdk//:mksd",
)

genrule(
    name = "empty_snapshot_fs",
    srcs = ["snapshots.img.zip"],
    outs = ["snapshots.img"],
    tools = ["@bazel_tools//tools/zip:zipper"],
    cmd = "$(location @bazel_tools//tools/zip:zipper) x $< -d $(@D)",
)

filegroup(
    name = "xvfb_support",
    srcs = [
        "@androidsdk//:emulator_shared_libs",
    ],
)

alias(
    name = "sdk_path",
    actual = "@androidsdk//:sdk_path",
)

# Unified launcher is not included with Bazel or the Android SDK. Instead, it
# comes from an external repository that must be set up in order for
# android_device to work.
alias(
    name = "unified_launcher",
    actual = "@android_test_support//tools/android/emulator:unified_launcher_head",
)

filegroup(
    name = "shbase",
    srcs = ["googletest.sh"],
)
