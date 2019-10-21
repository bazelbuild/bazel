licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//absl:_build_defs.bzl", "py2py3_test_binary", "py2and3_test")

py_library(
    name = "absltest",
    srcs = ["absltest.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pretty_print_reporter",
        ":xml_reporter",
        "//absl:_collections_abc",
        "//absl:_enum_module",
        "//absl:app",
        "//absl/flags",
        "//absl/logging",
        "//absl/third_party/unittest3_backport",
        "@six_archive//:six",
    ],
)

py_library(
    name = "flagsaver",
    srcs = ["flagsaver.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//absl/flags",
        "@six_archive//:six",
    ],
)

py_library(
    name = "parameterized",
    srcs = ["parameterized.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":absltest",
        "//absl:_collections_abc",
        "@six_archive//:six",
    ],
)

py_library(
    name = "xml_reporter",
    srcs = ["xml_reporter.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pretty_print_reporter",
        "//absl/third_party/unittest3_backport",
        "@six_archive//:six",
    ],
)

py_library(
    name = "_bazelize_command",
    testonly = 1,
    srcs = ["_bazelize_command.py"],
    srcs_version = "PY2AND3",
    visibility = ["//:__subpackages__"],
    deps = [
        ":_bazel_selected_py3",  # buildcleaner: keep
        "//absl/flags",
    ],
)

py_library(
    name = "_bazel_selected_py3",
    testonly = 1,
    srcs = select({
        "//absl:py3_mode": ["_bazel_selected_py3.py"],
        "//absl:py2_mode": [],
    }),
    srcs_version = "PY2AND3",
)

py_library(
    name = "_pretty_print_reporter",
    srcs = ["_pretty_print_reporter.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//absl/third_party/unittest3_backport",
    ],
)

py2and3_test(
    name = "tests/absltest_filtering_test",
    size = "small",
    srcs = ["tests/absltest_filtering_test.py"],
    data = [":tests/absltest_filtering_test_helper"],
    srcs_version = "PY2AND3",
    deps = [
        ":_bazelize_command",
        ":absltest",
        ":parameterized",
        "//absl/logging",
    ],
)

py2py3_test_binary(
    name = "tests/absltest_filtering_test_helper",
    testonly = 1,
    srcs = ["tests/absltest_filtering_test_helper.py"],
    srcs_version = "PY2AND3",
    deps = [":absltest"],
)

py2and3_test(
    name = "tests/absltest_randomization_test",
    size = "medium",
    srcs = ["tests/absltest_randomization_test.py"],
    data = [":tests/absltest_randomization_testcase"],
    srcs_version = "PY2AND3",
    deps = [
        ":_bazelize_command",
        ":absltest",
        ":parameterized",
        "//absl/flags",
    ],
)

py2py3_test_binary(
    name = "tests/absltest_randomization_testcase",
    testonly = 1,
    srcs = ["tests/absltest_randomization_testcase.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":absltest",
    ],
)

py2and3_test(
    name = "tests/absltest_sharding_test",
    size = "small",
    srcs = ["tests/absltest_sharding_test.py"],
    data = [":tests/absltest_sharding_test_helper"],
    srcs_version = "PY2AND3",
    deps = [
        ":_bazelize_command",
        ":absltest",
        "//absl/flags",
    ],
)

py2py3_test_binary(
    name = "tests/absltest_sharding_test_helper",
    testonly = 1,
    srcs = ["tests/absltest_sharding_test_helper.py"],
    srcs_version = "PY2AND3",
    deps = [":absltest"],
)

py2and3_test(
    name = "tests/absltest_test",
    size = "small",
    srcs = ["tests/absltest_test.py"],
    data = [":tests/absltest_test_helper"],
    srcs_version = "PY2AND3",
    deps = [
        ":_bazelize_command",
        ":absltest",
        ":parameterized",
        "//absl/flags",
        "@six_archive//:six",
    ],
)

py2py3_test_binary(
    name = "tests/absltest_test_helper",
    testonly = 1,
    srcs = ["tests/absltest_test_helper.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":absltest",
        "//absl/flags",
    ],
)

py2and3_test(
    name = "tests/flagsaver_test",
    srcs = ["tests/flagsaver_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":absltest",
        ":flagsaver",
        "//absl/flags",
    ],
)

py2and3_test(
    name = "tests/parameterized_test",
    srcs = ["tests/parameterized_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":absltest",
        ":parameterized",
        "//absl:_collections_abc",
        "@six_archive//:six",
    ],
)

py2and3_test(
    name = "tests/xml_reporter_test",
    srcs = ["tests/xml_reporter_test.py"],
    data = [":tests/xml_reporter_helper_test"],
    srcs_version = "PY2AND3",
    deps = [
        ":_bazelize_command",
        ":absltest",
        ":parameterized",
        ":xml_reporter",
        "//absl/logging",
        "//absl/third_party/unittest3_backport",
        "@mock_archive//:mock",
        "@six_archive//:six",
    ],
)

py2py3_test_binary(
    name = "tests/xml_reporter_helper_test",
    testonly = 1,
    srcs = ["tests/xml_reporter_helper_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":absltest",
        "//absl/flags",
    ],
)
