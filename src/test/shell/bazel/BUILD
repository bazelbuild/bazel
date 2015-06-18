package(default_visibility = ["//visibility:private"])

genrule(
    name = "langtools-copy",
    testonly = 1,
    srcs = ["//tools/defaults:java_langtools"],
    outs = ["langtools.jar"],
    cmd = "cp $< $@",
)

filegroup(
    name = "test-deps",
    testonly = 1,
    srcs = [
        "test-setup.sh",
        "testenv.sh",
        ":langtools-copy",
        "//examples:srcs",
        "//external:bootclasspath",
        "//external:extdir",
        "//external:jdk-default",
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
    name = "bazel_rust_example_test",
    srcs = ["bazel_rust_example_test.sh"],
    data = [":test-deps"],
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
        "//src/java_tools/singlejar:srcs",
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
    name = "workspace_test",
    srcs = ["workspace_test.sh"],
    data = [
        ":test-deps",
        "//src/main/java/com/google/devtools/build/workspace:generate_workspace",
    ],
)

sh_test(
    name = "bazel_sandboxing_test",
    srcs = ["bazel_sandboxing_test.sh"],
    data = [":test-deps"],
    tags = ["manual"],  # Test is still flaky.
)

sh_test(
    name = "bazel_worker_test",
    srcs = ["bazel_worker_test.sh"],
    data = [":test-deps"],
)

test_suite(
    name = "all_tests",
    visibility = ["//visibility:public"],
)
