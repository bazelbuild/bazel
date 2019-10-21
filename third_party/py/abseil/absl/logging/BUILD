licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//absl:_build_defs.bzl", "py2py3_test_binary", "py2and3_test")

py_library(
    name = "logging",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":converter",
        "//absl/flags",
        "@six_archive//:six",
    ],
)

py_library(
    name = "converter",
    srcs = ["converter.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

py2and3_test(
    name = "tests/converter_test",
    size = "small",
    srcs = ["tests/converter_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converter",
        ":logging",
        "//absl/testing:absltest",
    ],
)

py2and3_test(
    name = "tests/logging_test",
    size = "small",
    srcs = ["tests/logging_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":logging",
        "//absl/flags",
        "//absl/testing:absltest",
        "//absl/testing:flagsaver",
        "//absl/testing:parameterized",
        "@mock_archive//:mock",
        "@six_archive//:six",
    ],
)

py2and3_test(
    name = "tests/log_before_import_test",
    srcs = ["tests/log_before_import_test.py"],
    main = "tests/log_before_import_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":logging",
        "//absl/testing:absltest",
        "@mock_archive//:mock",
    ],
)

py2and3_test(
    name = "tests/verbosity_flag_test",
    srcs = ["tests/verbosity_flag_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":logging",
        "//absl/flags",
        "//absl/testing:absltest",
    ],
)

py2py3_test_binary(
    name = "tests/logging_functional_test_helper",
    testonly = 1,
    srcs = ["tests/logging_functional_test_helper.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":logging",
        "//absl:app",
        "//absl/flags",
        "@mock_archive//:mock",
        "@six_archive//:six",
    ],
)

py2and3_test(
    name = "tests/logging_functional_test",
    size = "large",
    srcs = ["tests/logging_functional_test.py"],
    data = [":tests/logging_functional_test_helper"],
    shard_count = 50,
    srcs_version = "PY2AND3",
    deps = [
        ":logging",
        "//absl/flags",
        "//absl/testing:_bazelize_command",
        "//absl/testing:absltest",
        "//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)
