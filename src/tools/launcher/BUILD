load(":win_rules.bzl", "CC_17_COPT", "LINK_UNICODE", "win_cc_binary", "win_cc_library")

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/tools/launcher/util:srcs"],
    visibility = ["//src:__pkg__"],
)

config_setting(
    name = "msvc_compiler",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "msvc-cl"},
)

config_setting(
    name = "clang-cl_compiler",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "clang-cl"},
)

config_setting(
    name = "mingw-gcc_compiler",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "mingw-gcc"},
)

win_cc_binary(
    name = "launcher",
    srcs = ["launcher_main.cc"],
    copts = CC_17_COPT,
    features = ["fully_static_link"],
    linkopts = LINK_UNICODE,
    linkstatic = True,
    visibility = [
        "//src:__pkg__",
        "//tools/launcher:__pkg__",
    ],
    deps = [
        ":bash_launcher",
        ":java_launcher",
        ":launcher_base",
        ":python_launcher",
        "//src/tools/launcher/util",
        "//src/tools/launcher/util:data_parser",
    ],
)

win_cc_library(
    name = "launcher_base",
    srcs = ["launcher.cc"],
    hdrs = ["launcher.h"],
    copts = CC_17_COPT,
    deps = [
        "//src/main/cpp/util:filesystem",
        "//src/tools/launcher/util",
        "//src/tools/launcher/util:data_parser",
    ],
)

win_cc_library(
    name = "java_launcher",
    srcs = ["java_launcher.cc"],
    hdrs = ["java_launcher.h"],
    copts = CC_17_COPT,
    deps = [
        ":launcher_base",
        "//src/main/native/windows:lib-process",
    ],
)

win_cc_library(
    name = "python_launcher",
    srcs = ["python_launcher.cc"],
    hdrs = ["python_launcher.h"],
    copts = CC_17_COPT,
    deps = [
        ":launcher_base",
        "//src/main/native/windows:lib-process",
    ],
)

win_cc_library(
    name = "bash_launcher",
    srcs = ["bash_launcher.cc"],
    hdrs = ["bash_launcher.h"],
    copts = CC_17_COPT,
    deps = [":launcher_base"],
)
