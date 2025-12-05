load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_python//python:defs.bzl", "py_test")

licenses(["notice"])  # 3-clause BSD

package(
    default_visibility = [
        "//src:__subpackages__",
        "//tools/def_parser:__subpackages__",
    ],
)

cc_library(
    name = "def_parser_lib",
    srcs = ["def_parser.cc"],
    hdrs = ["def_parser.h"],
    tags = [
        "__DONT_DEPEND_ON_DEF_PARSER__",
    ],
    target_compatible_with = ["@platforms//os:windows"],
)

cc_binary(
    name = "def_parser",
    srcs = ["def_parser_main.cc"],
    tags = [
        "__DONT_DEPEND_ON_DEF_PARSER__",
    ],
    target_compatible_with = ["@platforms//os:windows"],
    deps = [
        ":def_parser_lib",
    ],
)

py_test(
    name = "def_parser_test",
    srcs = [
        "def_parser_test.py",
    ],
    data = [":def_parser"],
    main = "def_parser_test.py",
    target_compatible_with = ["@platforms//os:windows"],
    deps = ["//src/test/py/bazel:test_base"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//src:__pkg__",
        "//third_party:__pkg__",
    ],
)
