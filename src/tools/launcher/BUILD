load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//tools/res:win_res.bzl", "windows_resources")
load(":launcher_maker_test.bzl", "launcher_maker_test")
load(":win_rules.bzl", "win_cc_binary", "win_cc_library")

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/tools/launcher/util:srcs"],
    visibility = ["//src:__pkg__"],
)

windows_resources(
    name = "win_resources",
    rc_files = ["win_resources.rc"],
    resources = ["win_manifest.xml"],
)

win_cc_binary(
    name = "launcher",
    srcs = ["launcher_main.cc"],
    visibility = [
        "//src/prebuilt_tools:__pkg__",
        "//tools/launcher:__pkg__",
    ],
    deps = [
        ":bash_launcher",
        ":java_launcher",
        ":launcher_base",
        ":python_launcher",
        ":win_resources",
        "//src/tools/launcher/util",
        "//src/tools/launcher/util:data_parser",
    ],
)

win_cc_library(
    name = "launcher_base",
    srcs = ["launcher.cc"],
    hdrs = ["launcher.h"],
    deps = [
        "//src/main/cpp/util:filesystem",
        "//src/main/native/windows:lib-process",
        "//src/tools/launcher/util",
        "//src/tools/launcher/util:data_parser",
    ],
)

win_cc_library(
    name = "java_launcher",
    srcs = ["java_launcher.cc"],
    hdrs = ["java_launcher.h"],
    deps = [
        ":launcher_base",
        "//src/main/native/windows:lib-process",
        "//src/tools/launcher/util",
    ],
)

win_cc_library(
    name = "python_launcher",
    srcs = ["python_launcher.cc"],
    hdrs = ["python_launcher.h"],
    deps = [
        ":launcher_base",
        "//src/main/native/windows:lib-process",
    ],
)

win_cc_library(
    name = "bash_launcher",
    srcs = ["bash_launcher.cc"],
    hdrs = ["bash_launcher.h"],
    deps = [":launcher_base"],
)

cc_binary(
    name = "launcher_maker",
    srcs = ["launcher_maker.cc"],
    visibility = [
        "//src/prebuilt_tools:__pkg__",
        "//tools/launcher:__pkg__",
    ],
    deps = ["//src/main/cpp/util:filesystem"],
)

launcher_maker_test(name = "launcher_maker_test")

sh_test(
    name = "launcher_maker_sh_test",
    srcs = ["launcher_maker_sh_test.sh"],
    args = [
        "$(rootpath :launcher_maker)",
        "$(rootpath //tools/launcher:launcher_maker.sh)",
    ],
    data = [
        ":launcher_maker",
        "//tools/launcher:launcher_maker.sh",
    ],
    # launcher_maker.sh is only used on non-Windows execution platforms.
    target_compatible_with = select({
        "@platforms//os:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
)
