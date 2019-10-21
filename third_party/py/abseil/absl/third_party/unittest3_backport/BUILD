licenses(["notice"])  # New BSD, Python Software Foundation

exports_files(["LICENSE"])

load("//absl:_build_defs.bzl", "py2and3_test")

py_library(
    name = "unittest3_backport",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//absl/testing:__pkg__"],
    deps = [
        ":case",
        ":result",
    ],
)

py_library(
    name = "case",
    srcs = ["case.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@six_archive//:six",
    ],
)

py_library(
    name = "result",
    srcs = ["result.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@six_archive//:six",
    ],
)

py2and3_test(
    name = "tests/unittest3_backport_test",
    size = "small",
    srcs = ["tests/unittest3_backport_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//absl/testing:absltest",
        "//absl/testing:xml_reporter",
        "@mock_archive//:mock",
        "@six_archive//:six",
    ],
)
