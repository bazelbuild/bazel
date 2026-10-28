load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
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

config_setting(
    name = "windows_clang",
    constraint_values = ["@platforms//os:windows"],
    flag_values = {"@rules_cc//cc/compiler:compiler": "clang"},
    visibility = ["//visibility:private"],
)

config_setting(
    name = "windows_gcc",
    constraint_values = ["@platforms//os:windows"],
    flag_values = {"@rules_cc//cc/compiler:compiler": "gcc"},
    visibility = ["//visibility:private"],
)

config_setting(
    name = "windows_mingw_gcc",
    constraint_values = ["@platforms//os:windows"],
    flag_values = {"@rules_cc//cc/compiler:compiler": "mingw-gcc"},
    visibility = ["//visibility:private"],
)

win_cc_binary(
    name = "launcher",
    srcs = ["launcher_main.cc"],
    # MinGW requires -municode to use wmain as the entry point. The launcher is
    # copied next to the binary it launches, where the DLLs of the MinGW C++
    # runtime aren't available, so link the runtime statically. Neither applies
    # to MSVC builds, such as the prebuilt launcher shipped with Bazel on
    # Windows.
    linkopts = select({
        ":windows_clang": [
            "-municode",
            "-static",
        ],
        ":windows_gcc": [
            "-municode",
            "-static",
        ],
        ":windows_mingw_gcc": [
            "-municode",
            "-static",
        ],
        "//conditions:default": [],
    }),
    visibility = [
        "//src:__pkg__",
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
        "//src:__pkg__",
        "//tools/launcher:__pkg__",
    ],
    deps = ["//src/main/cpp/util:filesystem"],
)

launcher_maker_test(name = "launcher_maker_test")
