load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "blaze_server_startup_options",
    srcs = ["BlazeServerStartupOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "safe_request_logging",
    srcs = ["SafeRequestLogging.java"],
)

java_library(
    name = "keep_going_option",
    srcs = ["KeepGoingOption.java"],
    deps = ["//src/main/java/com/google/devtools/common/options"],
)

java_library(
    name = "keep_state_after_build_option",
    srcs = ["KeepStateAfterBuildOption.java"],
    deps = ["//src/main/java/com/google/devtools/common/options"],
)

java_library(
    name = "loading_phase_threads_option",
    srcs = ["LoadingPhaseThreadsOption.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:TestType",
        "//src/main/java/com/google/devtools/build/lib/util:resource_converter",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:flogger",
    ],
)

java_library(
    name = "blaze_command_result",
    srcs = [
        "BlazeCommandResult.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/server:idle_task",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/protobuf:command_server_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "build_event_streamer_utils",
    srcs = ["BuildEventStreamerUtils.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/protobuf:test_status_java_proto",
    ],
)

java_library(
    name = "command_dispatcher",
    srcs = [
        "CommandDispatcher.java",
    ],
    deps = [
        ":blaze_command_result",
        "//src/main/java/com/google/devtools/build/lib/server:idle_task",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util/io:io-proto",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//third_party:guava",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "command_line_event",
    srcs = ["CommandLineEvent.java"],
    deps = [
        ":blaze_server_startup_options",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/util:pair",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:command_line_java_proto",
        "//src/main/protobuf:option_filters_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "common_command_options",
    srcs = ["CommonCommandOptions.java"],
    deps = [
        ":command_line_event",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "command_line_path_factory",
    srcs = [
        "CommandLinePathFactory.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
    ],
)

java_library(
    name = "memory_pressure",
    srcs = [
        "GcThrashingDetector.java",
        "MemoryPressureEvent.java",
        "MemoryPressureOptions.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "test_summary_options",
    srcs = ["TestSummaryOptions.java"],
    deps = ["//src/main/java/com/google/devtools/common/options"],
)

java_library(
    name = "ui_options",
    srcs = ["UiOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "starlark_options_parser",
    srcs = ["StarlarkOptionsParser.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_option_converters",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:scope",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/util:pair",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "config_flag_definitions",
    srcs = [
        "ConfigFlagDefinitions.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "options_supplier",
    srcs = ["OptionsSupplier.java"],
    deps = ["//src/main/java/com/google/devtools/common/options"],
)

java_library(
    name = "blaze_service",
    srcs = ["BlazeService.java"],
    deps = [
        ":options_supplier",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "instrumentation_output",
    srcs = ["InstrumentationOutput.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildtool:build_result",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "runtime",
    srcs = glob(
        ["*.java"],
        exclude = [
            "BlazeCommandResult.java",
            "BlazeService.java",
            "BuildEventStreamerUtils.java",
            "CommandDispatcher.java",
            "CommandLineEvent.java",
            "CommandLinePathFactory.java",
            "CommonCommandOptions.java",
            "ConfigFlagDefinitions.java",
            "GcThrashingDetector.java",
            "KeepGoingOption.java",
            "KeepStateAfterBuildOption.java",
            "LoadingPhaseThreadsOption.java",
            "MemoryPressureEvent.java",
            "MemoryPressureOptions.java",
            "OptionsSupplier.java",
            "SafeRequestLogging.java",
            "StarlarkOptionsParser.java",
            "TestSummaryOptions.java",
            "UiOptions.java",
            "InstrumentationOutput.java",
        ],
    ),
    deps = [
        ":blaze_command_result",
        ":blaze_service",
        ":build_event_streamer_utils",
        ":command_dispatcher",
        ":command_line_event",
        ":command_line_path_factory",
        ":common_command_options",
        ":config_flag_definitions",
        ":instrumentation_output",
        ":keep_state_after_build_option",
        ":loading_phase_threads_option",
        ":memory_pressure",
        ":options_supplier",
        ":safe_request_logging",
        ":starlark_options_parser",
        ":test_summary_options",
        ":ui_options",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_execution_inactivity_event",
        "//src/main/java/com/google/devtools/build/lib/actions:action_output_directory_helper",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/actions:fileset_output_symlink",
        "//src/main/java/com/google/devtools/build/lib/actions:resource_manager",
        "//src/main/java/com/google/devtools/build/lib/actions:runfiles_tree",
        "//src/main/java/com/google/devtools/build/lib/actions:shared_action_event",
        "//src/main/java/com/google/devtools/build/lib/analysis:actions/abstract_file_write_action",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_phase_complete_event",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/analysis:build_info_event",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:no_build_event",
        "//src/main/java/com/google/devtools/build/lib/analysis:no_build_request_finished_event",
        "//src/main/java/com/google/devtools/build/lib/analysis:server_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/coverage_report_action_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/test_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis:view_creation_failed_exception",
        "//src/main/java/com/google/devtools/build/lib/analysis:workspace_status_action",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:additional_configuration_change_event",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_option_converters",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports",
        "//src/main/java/com/google/devtools/build/lib/buildtool:build_request",
        "//src/main/java/com/google/devtools/build/lib/buildtool:build_request_options",
        "//src/main/java/com/google/devtools/build/lib/buildtool:build_result",
        "//src/main/java/com/google/devtools/build/lib/buildtool:command_precomplete_event",
        "//src/main/java/com/google/devtools/build/lib/buildtool:execution_progress_receiver",
        "//src/main/java/com/google/devtools/build/lib/buildtool/buildevent",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/exec:bin_tools",
        "//src/main/java/com/google/devtools/build/lib/exec:execution_options",
        "//src/main/java/com/google/devtools/build/lib/exec:executor_builder",
        "//src/main/java/com/google/devtools/build/lib/exec:module_action_context_registry",
        "//src/main/java/com/google/devtools/build/lib/exec:single_build_file_cache",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_cache",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_strategy_registry",
        "//src/main/java/com/google/devtools/build/lib/exec:test_log_helper",
        "//src/main/java/com/google/devtools/build/lib/exec/local:options",
        "//src/main/java/com/google/devtools/build/lib/jni",
        "//src/main/java/com/google/devtools/build/lib/metrics:garbage-collection-metrics-util",
        "//src/main/java/com/google/devtools/build/lib/metrics/criticalpath",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:package_piece_identifier",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/pkgcache:package_options",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:collect_local_resource_usage",
        "//src/main/java/com/google/devtools/build/lib/profiler:google-auto-profiler-utils",
        "//src/main/java/com/google/devtools/build/lib/profiler:system_network_stats_service",
        "//src/main/java/com/google/devtools/build/lib/profiler/memory:allocationtracker",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/query2/common:options",
        "//src/main/java/com/google/devtools/build/lib/query2/engine",
        "//src/main/java/com/google/devtools/build/lib/query2/query/output",
        "//src/main/java/com/google/devtools/build/lib/runtime/events",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/proto:cgroups_info_java_proto",
        "//src/main/java/com/google/devtools/build/lib/server",
        "//src/main/java/com/google/devtools/build/lib/server:gc_and_interner_shrinking_idle_task",
        "//src/main/java/com/google/devtools/build/lib/server:idle_task",
        "//src/main/java/com/google/devtools/build/lib/server:install_base_garbage_collector",
        "//src/main/java/com/google/devtools/build/lib/server:pid_file_watcher",
        "//src/main/java/com/google/devtools/build/lib/server:shutdown_hooks",
        "//src/main/java/com/google/devtools/build/lib/server/signal",
        "//src/main/java/com/google/devtools/build/lib/skyframe:analysis_progress_receiver",
        "//src/main/java/com/google/devtools/build/lib/skyframe:aspect_key_creator",
        "//src/main/java/com/google/devtools/build/lib/skyframe:build_result_listener",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configuration_phase_started_event",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:default_syscall_cache",
        "//src/main/java/com/google/devtools/build/lib/skyframe:diff_awareness",
        "//src/main/java/com/google/devtools/build/lib/skyframe:execution_finished_event",
        "//src/main/java/com/google/devtools/build/lib/skyframe:loading_phase_started_event",
        "//src/main/java/com/google/devtools/build/lib/skyframe:package_progress_receiver",
        "//src/main/java/com/google/devtools/build/lib/skyframe:package_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:precomputed_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:repository_mapping_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyfocus_options",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_cluster",
        "//src/main/java/com/google/devtools/build/lib/skyframe:target_pattern_phase_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:top_level_aspects_identified_event",
        "//src/main/java/com/google/devtools/build/lib/skyframe:top_level_status_events",
        "//src/main/java/com/google/devtools/build/lib/skyframe:workspace_info",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:event_listener",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:remote_analysis_caching_services_supplier",
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:TestType",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:custom_exit_code_publisher",
        "//src/main/java/com/google/devtools/build/lib/util:custom_failure_detail_publisher",
        "//src/main/java/com/google/devtools/build/lib/util:debug-logger-configurator",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:file_system_lock",
        "//src/main/java/com/google/devtools/build/lib/util:interrupted_failure_details",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/util:thread_dump_analyzer",
        "//src/main/java/com/google/devtools/build/lib/util:thread_dumper",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/util/io:io-proto",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/versioning:long_version_getter",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:output_service",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/worker:worker_process_metrics",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/main/protobuf:action_cache_java_proto",
        "//src/main/protobuf:command_server_java_proto",
        "//src/main/protobuf:crash_debugging_java_proto",
        "//src/main/protobuf:execution_graph_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/main/protobuf:memory_pressure_java_proto",
        "//src/main/protobuf:test_status_java_proto",
        "//third_party:auto_value",
        "//third_party:caffeine",
        "//third_party:error_prone_annotations",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "@com_google_protobuf//:protobuf_java",
        "@com_google_protobuf//:protobuf_java_util",
        "@zstd-jni",
    ],
)
