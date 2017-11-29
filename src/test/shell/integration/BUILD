package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/shell:__pkg__"],
)

filegroup(
    name = "test-deps",
    testonly = 1,
    srcs = [
        "//src/test/shell/bazel:test-deps",
    ],
)

sh_test(
    name = "progress_reporting_test",
    size = "large",
    srcs = ["progress_reporting_test.sh"],
    data = [":test-deps"],
    shard_count = 8,
)

sh_test(
    name = "runfiles_test",
    size = "medium",
    srcs = ["runfiles_test.sh"],
    data = [":test-deps"],
    tags = ["local"],
)

sh_test(
    name = "python_test",
    size = "medium",
    srcs = ["python_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "loading_phase_tests",
    size = "large",
    srcs = ["loading_phase_tests.sh"],
    data = [":test-deps"],
    shard_count = 4,
)

sh_test(
    name = "execution_phase_tests",
    size = "large",
    srcs = ["execution_phase_tests.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "experimental_ui_test",
    size = "large",
    srcs = ["experimental_ui_test.sh"],
    data = [":test-deps"],
    shard_count = 11,
)

sh_test(
    name = "java_integration_test",
    size = "large",
    srcs = ["java_integration_test.sh"],
    args = ["$(JAVABASE)"],
    data = [
        ":test-deps",
        "//src/test/shell:shell_utils",
    ],
    shard_count = 5,
    tags = ["jdk8"],
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
)

sh_test(
    name = "startup_options_test",
    size = "medium",
    srcs = ["startup_options_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "run_test",
    size = "medium",
    srcs = ["run_test.sh"],
    data = [":test-deps"],
    shard_count = 3,
)

sh_test(
    name = "stub_finds_runfiles_test",
    size = "medium",
    srcs = ["stub_finds_runfiles_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "discard_analysis_cache_test",
    size = "medium",
    srcs = ["discard_analysis_cache_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "output_filter_test",
    size = "large",
    srcs = ["output_filter_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_testjobs_test",
    srcs = ["bazel_testjobs_test.sh"],
    data = [":test-deps"],
    shard_count = 3,
)

sh_test(
    name = "bazel_query_test",
    size = "large",
    srcs = ["bazel_query_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "bazel_command_log_test",
    size = "medium",
    srcs = ["bazel_command_log_test.sh"],
    data = [":test-deps"],
    shard_count = 4,
)

sh_test(
    name = "bazel_javabase_test",
    size = "medium",
    srcs = ["bazel_javabase_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "client_sigint_test",
    size = "medium",
    srcs = ["client_sigint_test.sh"],
    data = [":test-deps"],
    # This test doesn't work with the sandbox on, see the source file
    # for details.
    tags = ["local"],
)

sh_test(
    name = "force_delete_output_test",
    size = "medium",
    srcs = ["force_delete_output_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "incompatible_changes_conflict_test",
    size = "small",
    srcs = ["incompatible_changes_conflict_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "skylark_flag_test",
    size = "medium",
    srcs = ["skylark_flag_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "ui_test",
    size = "medium",
    srcs = ["ui_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "rc_options_test",
    size = "medium",
    srcs = ["rc_options_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "cpp_test",
    size = "large",
    srcs = ["cpp_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "action_env_test",
    size = "large",
    srcs = ["action_env_test.sh"],
    data = [":test-deps"],
)

sh_library(
    name = "discard_graph_edges_lib",
    testonly = 1,
    srcs = ["discard_graph_edges_lib.sh"],
)

sh_test(
    name = "stamping_test",
    size = "medium",
    srcs = ["stamping_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "discard_graph_edges_test",
    size = "medium",
    srcs = ["discard_graph_edges_test.sh"],
    data = [
        ":discard_graph_edges_lib.sh",
        ":test-deps",
    ],
    shard_count = 6,
)

sh_test(
    name = "build_event_stream_test",
    size = "medium",
    srcs = ["build_event_stream_test.sh"],
    data = [":test-deps"],
    shard_count = 2,
)

sh_test(
    name = "bazel_worker_test",
    size = "large",
    srcs = ["bazel_worker_test.sh"],
    args = [
        "--worker_sandboxing=no",
        "non-sandboxed",
    ],
    data = [
        ":test-deps",
        "//src/test/java/com/google/devtools/build/lib:ExampleWorker_deploy.jar",
    ],
    shard_count = 3,
    tags = ["jdk8"],
)

sh_test(
    name = "bazel_sandboxed_worker_test",
    size = "large",
    srcs = ["bazel_worker_test.sh"],
    args = [
        "--worker_sandboxing",
        "sandboxed",
    ],
    data = [
        ":test-deps",
        "//src/test/java/com/google/devtools/build/lib:ExampleWorker_deploy.jar",
    ],
    # TODO(b/34450149): remove when no longer flaky
    flaky = 1,
    shard_count = 3,
    tags = ["jdk8"],
)

sh_test(
    name = "logging_test",
    size = "medium",
    srcs = ["logging_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "test_test",
    srcs = ["test_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "outputs_test",
    size = "medium",
    srcs = ["outputs_test.sh"],
    data = [":test-deps"],
)

sh_test(
    name = "process_wrapper_test",
    size = "medium",
    srcs = ["process-wrapper_test.sh"],
    data = [
        ":spend_cpu_time",
        ":test-deps",
        "//src/main/protobuf:execution_statistics.proto",
    ],
)

cc_binary(
    name = "spend_cpu_time",
    testonly = 1,
    srcs = ["spend_cpu_time.cc"],
    malloc = "//base:system_malloc",
)

########################################################################
# Test suites.

test_suite(
    name = "all_tests",
    visibility = ["//visibility:public"],
)
