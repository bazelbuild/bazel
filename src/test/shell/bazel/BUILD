package(default_visibility = ["//visibility:private"])

genrule(
    name = "langtools-copy",
    testonly = 1,
    srcs = ["//tools/defaults:java_langtools"],
    outs = ["langtools.jar"],
    cmd = "cp $< $@",
)

config_setting(
    name = "darwin",
    values = {"host_cpu": "darwin"},
)

filegroup(
    name = "objc-deps",
    testonly = 1,
    srcs = select({
        ":darwin": [
            "//src/objc_tools/bundlemerge:bundlemerge_deploy.jar",
            "//src/objc_tools/plmerge:plmerge_deploy.jar",
            "//src/objc_tools/xcodegen:xcodegen_deploy.jar",
            "//src/tools/xcode/actoolwrapper",
            "//src/tools/xcode/environment:environment_plist",
            "//src/tools/xcode/ibtoolwrapper",
            "//src/tools/xcode/momcwrapper",
            "//src/tools/xcode/realpath",
            "//src/tools/xcode/stdredirect:StdRedirect.dylib",
            "//src/tools/xcode/swiftstdlibtoolwrapper",
            "//third_party/iossim",
        ],
        "//conditions:default": [],
    }),
)

filegroup(
    name = "test-deps",
    testonly = 1,
    srcs = [
        "bazel_sandboxing_test_utils.sh",
        "remote_helpers.sh",
        "test-setup.sh",
        "testenv.sh",
        "testing_server.py",
        ":langtools-copy",
        ":objc-deps",
        "//examples:srcs",
        "//external:bootclasspath",
        "//external:extdir",
        "//external:jdk-default",
        "//src:bazel",
        "//src/java_tools/buildjar:JavaBuilder_deploy.jar",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass:GenClass_deploy.jar",
        "//src/java_tools/singlejar:SingleJar_deploy.jar",
        "//src/main/tools:namespace-sandbox",
        "//src/main/tools:process-wrapper",
        "//src/test/shell:bashunit",
        "//third_party:srcs",
        "//third_party/ijar",
        "//third_party/java/jdk/langtools:srcs",
        "//tools:srcs",
    ],
    visibility = [
        "//src/test/shell/bazel:__subpackages__",
    ],
)

sh_test(
    name = "bazel_example_test",
    size = "large",
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
    name = "bazel_objc_test",
    srcs = ["bazel_objc_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_execute_testlog",
    srcs = ["bazel_execute_testlog.sh"],
    data = [":test-deps"],
)

# TODO(bazel-team): zip is non-deterministic because of file timestamp,
# we should use a custom zip version (SingleJar?).
genrule(
    name = "doc-srcs",
    testonly = 1,
    srcs = [
        "//src/java_tools/singlejar:srcs",
        "//src/main/protobuf:srcs",
        "//src/main/java:srcs",
        "//src/main/java/com/google/devtools/build/lib/actions:srcs",
        "//src/main/java/com/google/devtools/build/lib/sandbox:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/genquery:srcs",
        "//src/main/java/com/google/devtools/build/lib/worker:srcs",
        "//src/main/java/com/google/devtools/build/lib/standalone:srcs",
        "//src/tools/xcode-common:srcs",
        "//third_party:srcs",
    ],
    outs = ["doc-srcs.zip"],
    cmd = "zip -q $@ $(SRCS)",
)

sh_test(
    name = "bazel_docgen_test",
    size = "large",
    srcs = ["bazel_docgen_test.sh"],
    data = [
        ":doc-srcs",
        ":test-deps",
    ],
)

sh_test(
    name = "external_integration_test",
    size = "large",
    srcs = ["external_integration_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "git_repository_test",
    size = "medium",
    srcs = ["git_repository_test.sh"],
    data = [
        ":test-deps",
        "//src/test/shell/bazel/testdata:git-repos",
    ],
)

sh_test(
    name = "local_repository_test",
    size = "large",
    srcs = ["local_repository_test.sh"],
    data = [":test-deps"],
    shard_count = 3,
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
    name = "maven_test",
    size = "large",
    srcs = ["maven_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "generate_workspace_test",
    srcs = ["generate_workspace_test.sh"],
    data = [
        ":test-deps",
        "//src/tools/generate_workspace",
    ],
    tags = ["manual"],  # Flaky, see issue #432
)

sh_test(
    name = "workspace_test",
    srcs = ["workspace_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "process_wrapper_test",
    size = "large",
    srcs = ["process-wrapper_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "namespace_runner_test",
    srcs = ["namespace-runner_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_sandboxing_test",
    srcs = ["bazel_sandboxing_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_sandboxing_cpp_test",
    srcs = ["bazel_sandboxing_cpp_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_worker_test",
    size = "large",
    srcs = ["bazel_worker_test.sh"],
    data = [
        ":test-deps",
        "//src/test/java:worker-example_deploy.jar",
    ],
)

sh_test(
    name = "bazel_workspace_status_test",
    srcs = ["bazel_workspace_status_test.sh"],
    data = [":test-deps"],
)

test_suite(
    name = "all_tests",
    visibility = ["//visibility:public"],
)
