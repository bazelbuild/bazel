# Description:
#  Authoritative shell tests for Bazel (non-blaze).

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_shell//shell:sh_library.bzl", "sh_library")
load("@rules_shell//shell:sh_test.bzl", "sh_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

sh_library(
    name = "test-deps",
    testonly = 1,
    srcs = ["//src/test/shell/bazel:test-deps"],
)

sh_test(
    name = "action_aspect_test",
    size = "medium",
    timeout = "long",
    srcs = ["action_aspect_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    tags = ["no_windows"],
)

sh_test(
    name = "action_env_test",
    size = "medium",
    srcs = ["action_env_test.sh"],
    data = [
        ":test-deps",
    ],
    shard_count = 8,
    tags = ["no_windows"],
)

sh_test(
    name = "analysis_phase_options_test",
    size = "medium",
    srcs = ["analysis_phase_options_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "analysis_test_test",
    size = "medium",
    srcs = ["analysis_test_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "aquery_test",
    size = "medium",
    timeout = "long",
    srcs = ["aquery_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "aspect_test",
    size = "large",
    srcs = ["aspect_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "bash_runfiles_v1_test",
    srcs = ["bash_runfiles_v1_test.sh"],
    data = ["BUILD"],
    deps = ["@bazel_tools//tools/bash/runfiles"],
)

sh_test(
    name = "bazel_command_log_test",
    size = "medium",
    srcs = ["bazel_command_log_test.sh"],
    data = [
        ":test-deps",
    ],
    shard_count = 4,
    tags = ["no_windows"],
)

sh_test(
    name = "bazel_hardened_sandboxed_worker_test",
    size = "medium",
    timeout = "long",
    srcs = ["bazel_worker_test.sh"],
    args = [
        "'--worker_sandboxing --experimental_worker_sandbox_hardening'",
        "sandboxed",
        "proto",
    ],
    data = [
        ":test-deps",
        "//src/test/java/com/google/devtools/build/lib/worker:ExampleWorker_deploy.jar",
    ],
    shard_count = 10,
    tags = [
        # MacOS does not have cgroups so it can't support hardened sandbox
        "no_macos",
        "no_windows",
    ],
)

sh_test(
    name = "bazel_javabase_test",
    size = "medium",
    srcs = ["bazel_javabase_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "bazel_json_worker_test",
    size = "medium",
    timeout = "long",
    srcs = ["bazel_worker_test.sh"],
    args = [
        "--worker_sandboxing=no",
        "non-sandboxed",
        "json",
    ],
    data = [
        ":test-deps",
        "//src/test/java/com/google/devtools/build/lib/worker:ExampleWorker_deploy.jar",
    ],
    shard_count = 7,
    tags = [
        "no_windows",
    ],
)

sh_test(
    name = "bazel_query_test",
    size = "medium",
    timeout = "long",
    srcs = ["bazel_query_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 5,
    tags = ["no_windows"],
)

sh_test(
    name = "bazel_sandboxed_worker_test",
    size = "medium",
    timeout = "long",
    srcs = ["bazel_worker_test.sh"],
    args = [
        "--worker_sandboxing",
        "sandboxed",
        "proto",
    ],
    data = [
        ":test-deps",
        "//src/test/java/com/google/devtools/build/lib/worker:ExampleWorker_deploy.jar",
    ],
    shard_count = 7,
    tags = [
        "no_windows",
    ],
)

sh_test(
    name = "bazel_sandboxed_worker_with_cgroups_test",
    size = "medium",
    timeout = "long",
    srcs = ["bazel_worker_test.sh"],
    args = [
        "'--worker_sandboxing --experimental_worker_use_cgroups_on_linux'",
        "sandboxed",
        "proto",
    ],
    data = [
        ":test-deps",
        "//src/test/java/com/google/devtools/build/lib/worker:ExampleWorker_deploy.jar",
    ],
    shard_count = 10,
    tags = [
        # MacOS does not have cgroups so it can't support hardened sandbox
        "no_macos",
        "no_windows",
    ],
)

sh_test(
    name = "bazel_testjobs_test",
    srcs = [
        "bazel_testjobs_test.sh",
    ],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 2,
)

sh_test(
    name = "bazel_worker_multiplexer_test",
    size = "medium",
    timeout = "long",
    srcs = ["bazel_worker_multiplexer_test.sh"],
    args = [
        "--worker_sandboxing=no",
        "non-sandboxed",
    ],
    data = [
        ":test-deps",
        "//src/test/java/com/google/devtools/build/lib/worker:ExampleWorkerMultiplexer_deploy.jar",
    ],
    shard_count = 5,
    tags = [
        "no_windows",
    ],
)

sh_test(
    name = "bazel_worker_test",
    size = "medium",
    timeout = "long",
    srcs = ["bazel_worker_test.sh"],
    args = [
        "--worker_sandboxing=no",
        "non-sandboxed",
        "proto",
    ],
    data = [
        ":test-deps",
        "//src/test/java/com/google/devtools/build/lib/worker:ExampleWorker_deploy.jar",
    ],
    shard_count = 7,
    tags = [
        "no_windows",
    ],
)

sh_test(
    name = "bazel_worker_with_cgroups_test",
    size = "medium",
    timeout = "long",
    srcs = ["bazel_worker_test.sh"],
    args = [
        "'--worker_sandboxing=no --experimental_worker_use_cgroups_on_linux'",
        "non-sandboxed",
        "proto",
    ],
    data = [
        ":test-deps",
        "//src/test/java/com/google/devtools/build/lib/worker:ExampleWorker_deploy.jar",
    ],
    shard_count = 7,
    tags = [
        # MacOS does not have cgroups so it can't support hardened sandbox
        "no_macos",
        "no_windows",
    ],
)

sh_test(
    name = "build_event_stream_test",
    size = "medium",
    timeout = "long",
    srcs = ["build_event_stream_test.sh"],
    data = [
        ":test-deps",
    ],
    shard_count = 10,
    tags = ["no_windows"],
)

sh_test(
    name = "builtins_injection_test",
    size = "medium",
    srcs = ["builtins_injection_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "client_sigint_test",
    size = "medium",
    srcs = ["client_sigint_test.sh"],
    data = [
        ":test-deps",
    ],
    tags = [
        "no-sandbox",
        "no_windows",
    ],
)

sh_test(
    name = "client_test",
    size = "medium",
    timeout = "long",
    srcs = ["client_test.sh"],
    data = [
        ":test-deps",
    ],
    shard_count = 8,
    tags = ["no_windows"],
)

sh_library(
    name = "config_stripped_outputs_lib",
    testonly = 1,
    srcs = ["config_stripped_outputs_lib.sh"],
)

sh_test(
    name = "config_stripped_outputs_test",
    size = "medium",
    timeout = "long",
    srcs = ["config_stripped_outputs_test.sh"],
    data = [
        ":config_stripped_outputs_lib",
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    tags = [
        # Bazel test requires sandboxing
        "no_windows",
    ],
)

sh_test(
    name = "configured_query_test",
    size = "large",
    srcs = ["configured_query_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 5,
)

# Copy protoc into a known location, since //third_party/protobuf:protoc
# might be an alias.  This is referenced from testenv.sh.
genrule(
    name = "copy_protoc",
    srcs = ["@com_google_protobuf//:protoc"],
    outs = ["protoc"],
    cmd = "cp $< $@",
)

sh_test(
    name = "cpp_test",
    size = "medium",
    timeout = "long",
    srcs = ["cpp_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "discard_analysis_cache_test",
    size = "medium",
    timeout = "long",
    srcs = ["discard_analysis_cache_test.sh"],
    args = ["$(JAVABASE)"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 3,
    tags = ["no_windows"],
    toolchains = ["@rules_java//toolchains:current_java_runtime"],
)

sh_library(
    name = "discard_graph_edges_lib",
    testonly = 1,
    srcs = ["discard_graph_edges_lib.sh"],
)

sh_test(
    name = "discard_graph_edges_test",
    size = "medium",
    timeout = "long",
    srcs = ["discard_graph_edges_test.sh"],
    args = ["$(JAVABASE)"],
    data = [
        ":discard_graph_edges_lib",
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 8,
    tags = ["no_windows"],
    toolchains = ["@rules_java//toolchains:current_java_runtime"],
)

sh_test(
    name = "dump_test",
    size = "medium",
    srcs = ["dump_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 1,
)

sh_test(
    name = "exec_group_test",
    timeout = "long",
    srcs = ["exec_group_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 5,
)

sh_test(
    name = "execution_strategies_test",
    size = "medium",
    srcs = ["execution_strategies_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "flags_scoping_integration_tests",
    srcs = ["flags_scoping_integration_tests.sh"],
    data = [
        ":test-deps",
        "//src/main/protobuf/project:project_proto_scl_filegroup",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "flagset_test",
    size = "large",
    srcs = ["flagset_test.sh"],
    data = [
        ":test-deps",
        "//src/main/protobuf/project:project_proto_scl_filegroup",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 3,
)

sh_test(
    name = "focus_test",
    size = "medium",
    timeout = "long",
    srcs = ["focus_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 4,
    tags = ["manual"],
)

sh_test(
    name = "force_delete_output_test",
    size = "medium",
    srcs = ["force_delete_output_test.sh"],
    data = [
        ":test-deps",
    ],
    tags = ["no_windows"],
)

sh_test(
    name = "info_test",
    srcs = ["info_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "java_integration_test",
    size = "medium",
    timeout = "long",
    srcs = ["java_integration_test.sh"],
    args = ["$(JAVABASE)"],
    data = [
        "java_integration_test_utils.sh",
        ":test-deps",
        "//src/test/shell:shell_utils",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 6,
    tags = ["no_windows"],
    toolchains = ["@rules_java//toolchains:current_java_runtime"],
)

sh_test(
    name = "java_sandboxing_test",
    size = "medium",
    srcs = ["java_sandboxing_test.sh"],
    data = [
        ":test-deps",
    ],
    shard_count = 4,
    # These are CPU-heavy due to running a large amount of analysis.
    tags = [
        "cpu:4",
        "no_windows",
    ],
)

sh_test(
    name = "jvm_flags_escaping_test",
    timeout = "long",
    srcs = ["jvm_flags_escaping_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "linux_sandbox_test",
    size = "medium",
    srcs = ["linux-sandbox_test.sh"],
    data = [
        ":execution_statistics_utils.sh",
        ":protoc",
        ":spend_cpu_time",
        ":test-deps",
        "//src/main/protobuf:execution_statistics.proto",
        "//src/main/tools:linux-sandbox",
        "//src/test/shell:sandboxing_test_utils",
    ],
    shard_count = 2,
    tags = [
        "no_windows",
    ],
)

sh_test(
    name = "loading_phase_posix_tests",
    size = "medium",
    srcs = ["loading_phase_posix_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    tags = [
        "no_windows",  # test uses symlinks and chmod
    ],
)

sh_test(
    name = "loading_phase_tests",
    size = "medium",
    timeout = "long",
    srcs = ["loading_phase_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 8,
    tags = [
        "requires-network",  # For Bzlmod
    ],
)

sh_test(
    name = "modify_execution_info_test",
    size = "medium",
    srcs = ["modify_execution_info_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
        "@zlib",
    ],
    shard_count = 5,
    tags = ["requires-network"],  # Allow this test to access internet to fetch protobuf dependencies.
)

sh_test(
    name = "nestedset_as_skykey_test",
    size = "medium",
    timeout = "long",
    srcs = ["nestedset_as_skykey_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 2,
)

sh_test(
    name = "nonincremental_builds_test",
    size = "medium",
    srcs = ["nonincremental_builds_test.sh"],
    args = ["$(JAVABASE)"],
    data = [
        ":discard_graph_edges_lib",
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    toolchains = ["@rules_java//toolchains:current_java_runtime"],
)

sh_test(
    name = "output_filter_test",
    size = "medium",
    timeout = "long",
    srcs = ["output_filter_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    tags = ["no_windows"],
)

sh_test(
    name = "outputs_test",
    size = "medium",
    srcs = ["outputs_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "param_file_flag_per_line_test",
    srcs = ["param_file_flag_per_line_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    tags = ["no_windows"],
)

sh_test(
    name = "platform_based_flags_test",
    srcs = ["platform_based_flags_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "platform_mapping_test",
    srcs = ["platform_mapping_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 5,
    tags = ["no_windows"],
)

sh_test(
    name = "process_wrapper_test",
    size = "medium",
    srcs = ["process-wrapper_test.sh"],
    data = [
        ":execution_statistics_utils.sh",
        ":protoc",
        ":spend_cpu_time",
        ":test-deps",
        "//src/main/protobuf:execution_statistics.proto",
        "//src/main/tools:process-wrapper",
    ],
    tags = ["no_windows"],
)

sh_test(
    name = "profiler_test",
    srcs = ["profiler_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "progress_reporting_test",
    size = "medium",
    srcs = ["progress_reporting_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 4,
)

sh_test(
    name = "python_stub_test",
    size = "medium",
    srcs = ["python_stub_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "python_test",
    size = "medium",
    timeout = "long",
    srcs = ["python_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "rc_options_test",
    size = "medium",
    srcs = ["rc_options_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "rule_transition_test",
    srcs = ["rule_transition_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    tags = ["no_windows"],
)

sh_test(
    name = "run_test",
    size = "medium",
    timeout = "long",
    srcs = ["run_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 6,
)

sh_test(
    name = "runfiles_test",
    size = "medium",
    timeout = "long",
    srcs = ["runfiles_test.sh"],
    data = [
        "runfiles_test_utils.sh",
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    tags = ["no-sandbox"],
)

sh_test(
    name = "sandboxing_test",
    size = "large",
    srcs = ["sandboxing_test.sh"],
    data = [
        ":test-deps",
        "//src/test/shell:sandboxing_test_utils",
    ],
    shard_count = 4,
    tags = ["no_windows"],
)

sh_test(
    name = "server_logging_test",
    size = "medium",
    srcs = ["server_logging_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

cc_binary(
    name = "spend_cpu_time",
    testonly = 1,
    srcs = select({
        "//src/conditions:windows": ["spend_cpu_time_windows.cc"],
        "//conditions:default": ["spend_cpu_time.cc"],
    }),
    visibility = [
        ":spend_cpu_time_users",
    ],
)

package_group(
    name = "spend_cpu_time_users",
    packages = [
        "//src/test/java/com/google/devtools/build/lib/exec/local/...",
        "//src/test/java/com/google/devtools/build/lib/sandbox/...",
        "//src/test/java/com/google/devtools/build/lib/shell/...",
    ],
)

sh_test(
    name = "starlark_configurations_test",
    size = "medium",
    srcs = ["starlark_configurations_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 5,
)

sh_test(
    name = "starlark_dependency_pruning_test",
    timeout = "long",
    srcs = ["starlark_dependency_pruning_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 2,
)

sh_test(
    name = "starlark_flag_test",
    size = "medium",
    srcs = ["starlark_flag_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "startup_options_test",
    size = "medium",
    srcs = ["startup_options_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "stub_finds_runfiles_local_test",
    size = "medium",
    timeout = "long",
    srcs = ["stub_finds_runfiles_test.sh"],
    args = [
        "standalone",
        "local",
    ],
    data = [
        ":test-deps",
    ],
    tags = ["no_windows"],
)

sh_test(
    name = "subrules_test",
    srcs = ["subrules_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "target_compatible_with_test",
    timeout = "long",
    srcs = ["target_compatible_with_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 6,
)

sh_test(
    name = "target_pattern_file_test",
    srcs = ["target_pattern_file_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "test_test",
    size = "medium",
    timeout = "long",
    srcs = ["test_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
)

sh_test(
    name = "thread_dump_test",
    srcs = ["thread_dump_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    tags = ["no_windows"],
)

sh_test(
    name = "toolchain_test",
    size = "large",
    srcs = ["toolchain_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 10,
    tags = ["no_windows"],
)

sh_test(
    name = "toolchain_transition_test",
    srcs = ["toolchain_transition_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    tags = ["no_windows"],
)

sh_test(
    name = "ui_test",
    size = "medium",
    timeout = "long",
    srcs = ["ui_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 11,
)

sh_test(
    name = "unicode_test",
    srcs = ["unicode_test.sh"],
    data = [
        "unicode_test.bzl",
        "unicode_test_BUILD",
        "unicode_test_expected.txt",
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    # TODO(arostovtsev): figure out how to make this test Windows-compatible.
    tags = ["no_windows"],
)

sh_test(
    name = "validation_actions_test",
    srcs = ["validation_actions_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    shard_count = 2,
    tags = ["no_windows"],
)

sh_test(
    name = "watchfs_test",
    srcs = ["watchfs_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    tags = ["no_windows"],
)

sh_test(
    name = "workspace_status_test",
    srcs = ["workspace_status_test.sh"],
    data = [
        ":test-deps",
        "@bazel_tools//tools/bash/runfiles",
    ],
    tags = ["no_windows"],
)

########################################################################
# Test suites.
test_suite(
    name = "small_tests",
    testonly = 1,
    tags = ["small"],
)

test_suite(
    name = "medium_tests",
    testonly = 1,
    tags = ["medium"],
)

# Test suites.
test_suite(
    name = "all_tests",
    visibility = ["//visibility:public"],
)
