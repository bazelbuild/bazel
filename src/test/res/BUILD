load("//src/main/res:win_res.bzl", "windows_resources")
load("@bazel_tools//tools/build_rules:test_rules.bzl", "file_test")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

cc_binary(
    name = "app",
    testonly = 1,
    srcs = [
        "app.cc",
        "app.h",
    ],
    linkopts = select({
        "//src/conditions:windows": ["-DEFAULTLIB:user32.lib"],
        "//conditions:default": [],
    }),
    deps = [":res"],
)

windows_resources(
    name = "res",
    testonly = 1,
    rc_files = ["app.rc"],
    resources = ["app.h"],
)

genrule(
    name = "run_app",
    testonly = 1,
    outs = ["app.out"],
    cmd = "$(location :app) > $@",
    tools = [":app"],
)

file_test(
    name = "res_test",
    content = select({
        "//src/conditions:windows": "l=12, p=(Hello string)",
        "//conditions:default": "not supported",
    }),
    file = ":run_app",
)

test_suite(name = "all_tests")

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
    visibility = ["//src:__pkg__"],
)
