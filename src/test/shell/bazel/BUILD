package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/test/shell/bazel/android:srcs",
        "//src/test/shell/bazel/apple:srcs",
        "//src/test/shell/bazel/testdata:srcs",
    ],
    visibility = ["//src/test/shell:__pkg__"],
)

genrule(
    name = "langtools-copy",
    testonly = 1,
    srcs = ["//third_party/java/jdk/langtools:javac_jar"],
    outs = ["langtools.jar"],
    cmd = "cp $< $@",
)

filegroup(
    name = "test-deps",
    testonly = 1,
    srcs = [
        "bazel_sandboxing_test_utils.sh",
        "remote_helpers.sh",
        "testing_server.py",
        ":langtools-copy",
        "//examples:srcs",
        "//src:bazel",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass:GenClass_deploy.jar",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner:Runner_deploy.jar",
        "//src/main/tools:linux-sandbox",
        "//src/main/tools:process-wrapper",
        "//src/test/shell:bashunit",
        "//src/test/shell:bin/bazel",
        "//src/test/shell:integration_test_setup.sh",
        "//src/test/shell:testenv.sh",
        "//third_party:srcs",
        "//third_party/ijar",
        "//third_party/java/jdk/langtools:test-srcs",
        "//tools:srcs",
        "@local_jdk//:bootclasspath",
        # TODO(cushon): migrate to extclasspath and delete
        "@local_jdk//:extdir",
        "@local_jdk//:jdk",
    ] + select({
        # TODO(bazel-team): Once https://github.com/bazelbuild/bazel/issues/2241
        # is resolved, use cc implementation of singlejar on windows
        "//src:windows": ["//src/java_tools/singlejar:SingleJar_deploy.jar"],
        "//src:windows_msys": ["//src/java_tools/singlejar:SingleJar_deploy.jar"],
        "//src:windows_msvc": ["//src/java_tools/singlejar:SingleJar_deploy.jar"],
        "//conditions:default": ["//src/tools/singlejar:singlejar"],
    }),
    visibility = [
        "//src/test/shell:__subpackages__",
    ],
)

sh_test(
    name = "bazel_example_test",
    size = "large",
    srcs = ["bazel_example_test.sh"],
    data = [
        ":test-deps",
        "//src/test/shell/bazel/apple:objc-deps",
    ],
    shard_count = 3,
)

sh_test(
    name = "bazel_windows_example_test",
    size = "large",
    srcs = ["bazel_windows_example_test.sh"],
    data = [
        ":test-deps",
    ],
)

sh_test(
    name = "cpp_darwin_integration_test",
    size = "large",
    srcs = ["cpp_darwin_integration_test.sh"],
    data = [
        ":test-deps",
    ],
)

sh_test(
    name = "bazel_tools_test",
    size = "large",
    srcs = ["bazel_tools_test.sh"],
    data = [
        ":test-deps",
        "//:workspace-file",
    ],
)

sh_test(
    name = "bazel_random_characters_test",
    size = "large",
    srcs = ["bazel_random_characters_test.sh"],
    data = [":test-deps"],
    tags = ["nowindows"],
)

sh_test(
    name = "bazel_java_test",
    size = "large",
    srcs = ["bazel_java_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "persistent_test_runner_test",
    size = "large",
    srcs = ["persistent_test_runner_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_build_event_stream_test",
    size = "medium",
    srcs = ["bazel_build_event_stream_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_experimental_ui_test",
    size = "medium",
    srcs = ["bazel_experimental_ui_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_rules_test",
    size = "large",
    srcs = ["bazel_rules_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_test_test",
    srcs = ["bazel_test_test.sh"],
    data = [":test-deps"],
    shard_count = 3,
    tags = ["local"],
)

sh_test(
    name = "bazel_coverage_test",
    srcs = ["bazel_coverage_test.sh"],
    data = [":test-deps"],
    tags = [
        "jdk8",  # TODO(elenairina): Re-enable for JDK7 once https://github.com/bazelbuild/bazel/issues/2890 is fixed.
        "local",
    ],
)

sh_test(
    name = "bazel_localtest_test",
    srcs = ["bazel_localtest_test.sh"],
    data = [":test-deps"],
    tags = ["local"],
)

sh_test(
    name = "bazel_execute_testlog",
    srcs = ["bazel_execute_testlog.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_toolchain_test",
    size = "medium",
    srcs = ["bazel_toolchain_test.sh"],
    data = [":test-deps"] + select({
        "//src:darwin": [],
        "//src:darwin_x86_64": [],
        "//src:windows": [],
        "//src:windows_msys": [],
        "//src:windows_msvc": [],
        "//conditions:default": ["//src/test/shell/bazel/testdata:bazel_toolchain_test_project_pkg"],
    }),
    tags = [
        "requires-network",
    ],
)

# TODO(bazel-team): zip is non-deterministic because of file timestamp,
# we should use a custom zip version (SingleJar?).
genrule(
    name = "doc-srcs",
    testonly = 1,
    srcs = [
        "//src/java_tools/singlejar:srcs",
        "//src/main/protobuf:srcs",
        "//src/main/java/com/google/devtools/build/lib:srcs",
        "//src/main/java/com/google/devtools/build/lib/actions:srcs",
        "//src/main/java/com/google/devtools/build/lib/sandbox:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/genquery:srcs",
        "//src/main/java/com/google/devtools/build/lib/worker:srcs",
        "//src/main/java/com/google/devtools/build/lib/standalone:srcs",
        "//src/tools/xcode-common:srcs",
        "//third_party:srcs",
    ],
    outs = ["doc-srcs.zip"],
    cmd = "echo $(SRCS) | tr ' ' '\n' | zip -q@ $@",
)

sh_test(
    name = "bazel_docgen_test",
    size = "large",
    srcs = ["bazel_docgen_test.sh"],
    data = ["//src/main/java/com/google/devtools/build/lib:gen_buildencyclopedia"],
)

sh_test(
    name = "external_correctness_test",
    size = "large",
    srcs = ["external_correctness_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "external_integration_test",
    size = "large",
    srcs = ["external_integration_test.sh"],
    data = [":test-deps"],
    shard_count = 3,
)

sh_test(
    name = "git_repository_test",
    size = "large",
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
    name = "local_repository_test_jdk8",
    size = "medium",
    srcs = ["local_repository_test_jdk8.sh"],
    data = [":test-deps"],
    tags = ["jdk8"],
)

sh_test(
    name = "cross_repository_test",
    size = "large",
    srcs = ["cross_repository_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "external_skylark_load_test",
    size = "large",
    srcs = ["external_skylark_load_test.sh"],
    data = [":test-deps"],
    shard_count = 6,
)

sh_test(
    name = "skylark_repository_test",
    size = "large",
    srcs = ["skylark_repository_test.sh"],
    data = [":test-deps"],
    shard_count = 6,
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

# To run this test, ensure that maven_dependency_plugin() and
# android_repositories() in WORKSPACE are uncommented.
sh_test(
    name = "maven_skylark_test",
    size = "medium",
    srcs = ["maven_skylark_test.sh"],
    data = [
        ":test-deps",
        "//external:android_ndk_for_testing",
        "//external:android_sdk_for_testing",
        "@m2//:files",
    ],
    tags = ["manual"],
)

sh_test(
    name = "workspace_test",
    size = "large",
    srcs = ["workspace_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "process_wrapper_test",
    size = "medium",
    srcs = ["process-wrapper_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "linux_sandbox_test",
    size = "large",
    srcs = ["linux-sandbox_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "cc_integration_test",
    size = "medium",
    srcs = ["cc_integration_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_sandboxing_test",
    size = "large",
    srcs = ["bazel_sandboxing_test.sh"],
    data = [":test-deps"],
    tags = ["local"],
)

sh_test(
    name = "bazel_sandboxing_cpp_test",
    srcs = ["bazel_sandboxing_cpp_test.sh"],
    data = [":test-deps"],
    tags = ["local"],
)

sh_test(
    name = "bazel_workspace_status_test",
    size = "large",
    srcs = ["bazel_workspace_status_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bound_targets_test",
    size = "large",
    srcs = ["bound_targets_test.sh"],
    data = [
        ":test-deps",
        "//:workspace-file",
        "//src/test/shell/bazel/apple:objc-deps",
    ],
)

sh_test(
    name = "remote_execution_test",
    size = "large",
    srcs = ["remote_execution_test.sh"],
    data = [
        ":test-deps",
        "//src/tools/remote_worker",
    ],
)

sh_test(
    name = "remote_execution_sandboxing_test",
    size = "large",
    srcs = ["remote_execution_sandboxing_test.sh"],
    data = [
        ":test-deps",
        "//src/tools/remote_worker",
    ],
)

sh_test(
    name = "client_test",
    size = "medium",
    srcs = ["client_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "execroot_test",
    size = "medium",
    srcs = ["execroot_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_repository_cache_test",
    size = "large",
    srcs = ["bazel_repository_cache_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_with_jdk_test",
    size = "medium",
    srcs = ["bazel_with_jdk_test.sh"],
    data = [
        ":test-deps",
        "//src:bazel_with_jdk",
    ],
)

sh_test(
    name = "build_files_test",
    size = "medium",
    srcs = ["build_files_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_bootstrap_distfile_test",
    timeout = "eternal",
    srcs = ["bazel_bootstrap_distfile_test.sh"],
    args = ["$(location //:bazel-distfile)"],
    data = [
        ":test-deps",
        "//:bazel-distfile",
    ],
    tags = ["jdk8"],
)

sh_test(
    name = "rule_test_test",
    size = "medium",
    srcs = ["rule_test_test.sh"],
    data = [":test-deps"],
    shard_count = 2,
)

sh_test(
    name = "help_test",
    size = "small",
    srcs = ["help_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "toolchain_test",
    size = "medium",
    srcs = ["toolchain_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "java_launcher_test",
    size = "medium",
    srcs = ["java_launcher_test.sh"],
    args = ["$(JAVABASE)"],
    data = [":test-deps"],
)

test_suite(
    name = "all_tests",
    visibility = ["//visibility:public"],
)
