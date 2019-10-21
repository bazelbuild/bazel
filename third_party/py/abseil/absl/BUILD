licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(":_build_defs.bzl", "py2py3_test_binary", "py2and3_test")

py_library(
    name = "app",
    srcs = [
        "app.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":command_name",
        "//absl/flags",
        "//absl/logging",
    ],
)

py_library(
    name = "command_name",
    srcs = ["command_name.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

# NOTE: Because of how Bazel's Python support works, its possible (quite likely,
# actually) to be running Python 3 without this config setting being matched,
# so be careful about assuming or requiring it to match for a target to
# correctly work under Python 3.
# See: https://github.com/bazelbuild/bazel/issues/4815
config_setting(
    name = "py3_mode",
    flag_values = {"@bazel_tools//tools/python:python_version": "PY3"},
    visibility = [":__subpackages__"],
)

config_setting(
    name = "py2_mode",
    flag_values = {"@bazel_tools//tools/python:python_version": "PY2"},
    visibility = [":__subpackages__"],
)

py_library(
    name = "_enum_module",
    srcs = ["_enum_module.py"],
    visibility = [":__subpackages__"],
    deps = [
        "@six_archive//:six",
    ] + select({
        # In some Python 3 environments, despite the _enum_module import path
        # tricks, the enum backport still ends up on sys.path before the
        # stdlib, which more-or-less breaks Python
        ":py3_mode": [],
        "//conditions:default": ["@enum34_archive//:enum"],
    }),
)

py_library(
    name = "_collections_abc",
    srcs = ["_collections_abc.py"],
    visibility = [":__subpackages__"],
    deps = ["@six_archive//:six"],
)

py_library(
    name = "tests/app_test_helper",
    testonly = 1,
    srcs = ["tests/app_test_helper.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":app",
        "//absl/flags",
    ],
)

py2py3_test_binary(
    name = "tests/app_test_helper_pure_python",
    testonly = 1,
    srcs = ["tests/app_test_helper.py"],
    main = "tests/app_test_helper.py",
    srcs_version = "PY2AND3",
    deps = [
        ":app",
        "//absl/flags",
    ],
)

py2and3_test(
    name = "tests/app_test",
    srcs = ["tests/app_test.py"],
    data = [":tests/app_test_helper_pure_python"],
    srcs_version = "PY2AND3",
    deps = [
        ":_enum_module",
        ":app",
        ":tests/app_test_helper",
        "//absl/flags",
        "//absl/testing:_bazelize_command",
        "//absl/testing:absltest",
        "//absl/testing:flagsaver",
        "@mock_archive//:mock",
        "@six_archive//:six",
    ],
)

py2and3_test(
    name = "tests/command_name_test",
    srcs = ["tests/command_name_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":command_name",
        "//absl/testing:absltest",
        "@mock_archive//:mock",
    ],
)
