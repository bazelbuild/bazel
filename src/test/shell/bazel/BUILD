package(default_visibility = ["//visibility:private"])

filegroup(
    name = "test-deps",
    testonly = 1,
    srcs = [
        "test-setup.sh",
        "testenv.sh",
        "//examples:srcs",
        "//external:bootclasspath",
        "//external:jdk-default",
        "//external:langtools",
        "//src:bazel",
        "//src/java_tools/buildjar:JavaBuilder_deploy.jar",
        "//src/java_tools/singlejar:SingleJar_deploy.jar",
        "//src/test/shell:bashunit",
        "//third_party:srcs",
        "//third_party/ijar",
        "//tools:srcs",
    ],
)

sh_test(
    name = "bazel_example_test",
    srcs = ["bazel_example_test.sh"],
    data = [":test-deps"],
    shard_count = 3,
)

sh_test(
    name = "bazel_rules_test",
    srcs = ["bazel_rules_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_test_test",
    srcs = ["bazel_test_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_execute_testlog",
    srcs = ["bazel_execute_testlog.sh"],
    data = [":test-deps"],
)

# TODO(bazel-team): zip is non-deterinistic because of file timestamp,
# we should use a custom zip version (SingleJar?).
genrule(
    name = "doc-srcs",
    testonly = 1,
    srcs = [
        "//src/main/protobuf:srcs",
        "//src/main/java:srcs",
        "//src/tools/xcode-common:srcs",
        "//third_party:srcs",
    ],
    outs = ["doc-srcs.zip"],
    cmd = "zip -q $@ $(SRCS)",
)

sh_test(
    name = "bazel_docgen_test",
    srcs = ["bazel_docgen_test.sh"],
    data = [
        ":doc-srcs",
        ":test-deps",
    ],
)

sh_test(
    name = "external_integration_test",
    size = "medium",
    srcs = ["external_integration_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "local_repository_test",
    size = "medium",
    srcs = ["local_repository_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "runfiles_test",
    size = "medium",
    srcs = ["runfiles_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "empty_package_test",
    srcs = ["empty_package.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "location_test",
    srcs = ["location_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_sandboxing_test",
    srcs = ["bazel_sandboxing_test.sh"],
    data = [":test-deps"],
    tags = ["manual"],
)

test_suite(
    name = "all_tests",
    visibility = ["//visibility:public"],
)
