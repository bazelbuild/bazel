load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

FILES = [
    "action_cache",
    "bazel_flags",
    "builtin",
    "crash_debugging",
    "crosstool_config",
    "deps",
    "desugar_deps",
    "execution_statistics",
    "extra_actions_base",
    "java_compilation",
    "memory_pressure",
    "strategy_policy",
    "test_status",
    "worker_protocol",
    "execution_graph",
    "file_invalidation_data",
]

[proto_library(
    name = s + "_proto",
    srcs = [s + ".proto"],
) for s in FILES]

proto_library(
    name = "build_proto",
    srcs = ["build.proto"],
    deps = [":stardoc_output_proto"],
)

proto_library(
    name = "analysis_v2_proto",
    srcs = ["analysis_v2.proto"],
    deps = [":build_proto"],
)

proto_library(
    name = "command_server_proto",
    srcs = ["command_server.proto"],
    deps = [
        ":failure_details_proto",
        "@com_google_protobuf//:any_proto",
    ],
)

proto_library(
    name = "failure_details_proto",
    srcs = ["failure_details.proto"],
    deps = ["@com_google_protobuf//:descriptor_proto"],
)

proto_library(
    name = "invocation_policy_proto",
    srcs = ["invocation_policy.proto"],
    deps = [":strategy_policy_proto"],
)

proto_library(
    name = "option_filters_proto",
    srcs = ["option_filters.proto"],
)

proto_library(
    name = "command_line_proto",
    srcs = ["command_line.proto"],
    deps = [":option_filters_proto"],
)

proto_library(
    name = "cache_salt_proto",
    srcs = ["cache_salt.proto"],
)

proto_library(
    name = "remote_scrubbing_proto",
    srcs = ["remote_scrubbing.proto"],
)

proto_library(
    name = "spawn_proto",
    srcs = ["spawn.proto"],
    deps = [
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:timestamp_proto",
    ],
)

proto_library(
    name = "stardoc_output_proto",
    srcs = ["stardoc_output.proto"],
)

proto_library(
    name = "xcode_proto",
    srcs = ["xcode_config.proto"],
)

# The following targets are not available in @bazel_tools
# to give a clear error message if someone tries to use them.

UNSUPPORTED_LANGUAGE_SPECIFIC_TARGETS = [
    "build_java_proto",
    "build_java_proto_srcs",
    "analysis_v2_py_proto",
    "analysis_v2_java_proto",
    "analysis_v2_java_proto_srcs",
    "any_java_proto",
    "wrappers_java_proto",
    "command_server_java_proto",
    "command_server_java_proto_srcs",
    "failure_details_java_proto",
    "failure_details_java_proto_srcs",
    "invocation_policy_java_proto",
    "invocation_policy_java_proto_srcs",
    "option_filters_java_proto",
    "option_filters_java_proto_srcs",
    "command_line_java_proto",
    "command_line_java_proto_srcs",
    "desugar_deps_cc_proto",
    "worker_protocol_cc_proto",
    "command_server_java_grpc",
    "command_server_cc_proto",
    "command_server_cc_grpc",
    "build_pb_py",
    "profile_java_proto_srcs",
    "execution_statistics_cc_proto",
    "remote_execution_log_proto",
    "remote_execution_log_java_proto",
    "remote_execution_log_java_proto_srcs",
    "cache_salt_java_proto",
    "cache_salt_java_proto_srcs",
    "remote_scrubbing_java_proto",
    "remote_scrubbing_java_proto_srcs",
    "bazel_output_service_java_proto",
    "bazel_output_service_java_proto_srcs",
    "bazel_output_service_cc_proto",
    "bazel_output_service_rev2_java_proto",
    "bazel_output_service_rev2_java_proto_srcs",
    "bazel_output_service_rev2_cc_proto",
    "bazel_output_service_java_grpc",
    "bazel_output_service_cc_grpc",
    "spawn_java_proto",
    "spawn_java_proto_srcs",
    "stardoc_output_java_proto",
    "stardoc_output_java_proto_srcs",
    "xcode_java_proto",
    "xcode_cc_proto",
    "xcode_java_proto_srcs",
]

[alias(
    name = s + "_java_proto",
    actual = ":ERROR_language_specific_proto_library_is_unsupported_in_bazel_tools",
) for s in FILES]

[alias(
    name = s + "_java_proto_srcs",
    actual = ":ERROR_language_specific_proto_library_is_unsupported_in_bazel_tools",
) for s in FILES]

[alias(
    name = name,
    actual = ":ERROR_language_specific_proto_library_is_unsupported_in_bazel_tools",
) for name in UNSUPPORTED_LANGUAGE_SPECIFIC_TARGETS]

genrule(
    name = "ERROR_language_specific_proto_library_is_unsupported_in_bazel_tools",
    outs = ["unused.txt"],
    cmd = "echo '\n\nLanguage specific proto targets are not available in @bazel_tools. Please use the proto_library and generate your own code.\n\n' >&2 && exit 1",
)

