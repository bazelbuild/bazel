package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

sh_test(
    name = "remote_execution_test",
    size = "large",
    timeout = "eternal",
    srcs = ["remote_execution_test.sh"],
    data = [
        "//src/test/shell/bazel:test-deps",
        "//src/tools/remote:worker",
    ],
)

sh_test(
    name = "remote_execution_http_test",
    size = "large",
    srcs = ["remote_execution_http_test.sh"],
    data = [
        "//src/test/shell/bazel:test-deps",
        "//src/tools/remote:worker",
    ],
)

sh_test(
    name = "remote_execution_sandboxing_test",
    size = "large",
    srcs = ["remote_execution_sandboxing_test.sh"],
    data = [
        "//src/test/shell:sandboxing_test_utils.sh",
        "//src/test/shell/bazel:test-deps",
        "//src/tools/remote:worker",
    ],
)

sh_test(
    name = "remote_execution_tls_test",
    size = "large",
    timeout = "eternal",
    srcs = ["remote_execution_tls_test.sh"],
    data = [
        "//src/test/shell/bazel:test-deps",
        "//src/test/testdata/test_tls_certificate",
        "//src/tools/remote:worker",
    ],
)
