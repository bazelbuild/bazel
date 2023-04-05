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
        "__DONT_DEPEND_ON_DEF_PARSER__"
    ],
)

cc_binary(
    name = "def_parser",
    srcs = ["def_parser_main.cc"],
    deps = [
        ":def_parser_lib",
    ],
    tags = [
        "__DONT_DEPEND_ON_DEF_PARSER__"
    ],
)

py_test(
    name = "def_parser_test",
    srcs = select({
        "//src/conditions:windows": ["def_parser_test.py"],
        "//conditions:default": ["empty_test.py"],
    }),
    data = select({
        "//src/conditions:windows": [":def_parser"],
        "//conditions:default": [],
    }),
    main = select({
        "//src/conditions:windows": "def_parser_test.py",
        "//conditions:default": "empty_test.py",
    }),
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
