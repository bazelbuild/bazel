load("@rules_java//java:defs.bzl", "java_library")

########################################################################
# Description:
#   Main Java code for Bazel
#
# IMPORTANT:  NOT A PUBLIC INTERFACE.
#
# DO NOT ADD A DEPENDENCY on any targets or sub-packages outside of
# __subpackages__ without first talking to Bazel maintainers.
#
# Here are two good reasons:
#
# 1.  FRAGILITY. This is not a stable API, it's an internal interface.
#     It changes from time to time, and your client *will* be broken.
#
# 2.  CORRECTNESS.  The BUILD file parser changes rapidly.  If
#     your application depends upon it, your application will start to
#     bit-rot almost immediately.
#
# There are better alternatives.  The 'bazel query' tool has
# stable command-line and programmatic interfaces that will work with any
# version of Bazel.  They are also debugged and documented, so you'll
# avoid many common traps and pitfalls when (e.g.)  computing
# dependencies.
package(
    default_visibility = ["//src:__subpackages__"],
)

# Generate list of all srcs via:
# bazel query 'filter("srcs", kind("filegroup rule", //src/main/java/com/google/devtools/build/lib/...))' | sort | sed -e "s/^/\"/" | sed -e "s/$/\",/" | fgrep -v "build/lib:srcs"
filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/docgen:srcs",
        "//src/main/java/com/google/devtools/build/lib/actions:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis:srcs",
        "//src/main/java/com/google/devtools/build/lib/authandtls:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel:srcs",
        "//src/main/java/com/google/devtools/build/lib/bugreport:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream:srcs",
        "//src/main/java/com/google/devtools/build/lib/causes:srcs",
        "//src/main/java/com/google/devtools/build/lib/clock:srcs",
        "//src/main/java/com/google/devtools/build/lib/cmdline:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashmap:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect:srcs",
        "//src/main/java/com/google/devtools/build/lib/concurrent:srcs",
        "//src/main/java/com/google/devtools/build/lib/dynamic:srcs",
        "//src/main/java/com/google/devtools/build/lib/events:srcs",
        "//src/main/java/com/google/devtools/build/lib/exec:srcs",
        "//src/main/java/com/google/devtools/build/lib/graph:srcs",
        "//src/main/java/com/google/devtools/build/lib/metrics:srcs",
        "//src/main/java/com/google/devtools/build/lib/network:srcs",
        "//src/main/java/com/google/devtools/build/lib/packages:srcs",
        "//src/main/java/com/google/devtools/build/lib/packages/metrics:srcs",
        "//src/main/java/com/google/devtools/build/lib/pkgcache:srcs",
        "//src/main/java/com/google/devtools/build/lib/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/profiler/callcounts:srcs",
        "//src/main/java/com/google/devtools/build/lib/profiler/memory:srcs",
        "//src/main/java/com/google/devtools/build/lib/profiler:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/common:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/compat:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/engine:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/query/aspectresolvers:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/query/output:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote:srcs",
        "//src/main/java/com/google/devtools/build/lib/repository:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules:srcs",
        "//src/main/java/com/google/devtools/build/lib/runtime/commands:srcs",
        "//src/main/java/com/google/devtools/build/lib/runtime/events:srcs",
        "//src/main/java/com/google/devtools/build/lib/runtime/mobileinstall:srcs",
        "//src/main/java/com/google/devtools/build/lib/sandbox:srcs",
        "//src/main/java/com/google/devtools/build/lib/server:srcs",
        "//src/main/java/com/google/devtools/build/lib/shell:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/module:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/server:srcs",
        "//src/main/java/net/starlark/java/annot/processor:srcs",
        "//src/main/java/net/starlark/java/annot:srcs",
        "//src/main/java/com/google/devtools/build/lib/ssd:srcs",
        "//src/main/java/com/google/devtools/build/lib/standalone:srcs",
        "//src/main/java/com/google/devtools/build/lib/supplier:srcs",
        "//src/main/java/com/google/devtools/build/lib/syntax:srcs",
        "//src/main/java/com/google/devtools/build/lib/unix:srcs",
        "//src/main/java/com/google/devtools/build/lib/unsafe:srcs",
        "//src/main/java/com/google/devtools/build/lib/util:srcs",
        "//src/main/java/com/google/devtools/build/lib/versioning:srcs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs:srcs",
        "//src/main/java/com/google/devtools/build/lib/vfs:srcs",
        "//src/main/java/com/google/devtools/build/lib/windows/jni:srcs",
        "//src/main/java/com/google/devtools/build/lib/windows:srcs",
        "//src/main/java/com/google/devtools/build/lib/worker:srcs",
        "//src/main/java/com/google/devtools/build/skyframe:srcs",
        "//src/main/java/com/google/devtools/common/options:srcs",
    ],
    visibility = [
        "//src:__pkg__",
        "//src/test/shell/bazel:__pkg__",
    ],
)

########################################################################
#
# The "foundation" library (concurrent, events, util, vfs, inmemoryfs, options)
# Filesystem abstraction, event reporting, options parsing, I/O routines.
#

# These packages apparently depend on :foundation. I have no idea why, but to
# avoid breaking them, I allow them to depend on it.
java_library(
    name = "foundation",
    exports = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

# Starlark language for configuration files.
#
# IMPORTANT:  NOT A PUBLIC INTERFACE. TARGETS SHOULDN'T DEPEND ON THIS.
#
java_library(
    name = "skylark-lang",
    visibility = ["//visibility:public"],
    exports = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
    ],
)

java_library(
    name = "syntax",
    srcs = [],
    visibility = ["//visibility:public"],
    exports = [
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
    ],
)

# Deprecated: This target has been moved. Depend on the actual target instead.
java_library(
    name = "shared-base-rules",
    exports = [
        "//src/main/java/com/google/devtools/build/lib/rules:nativedeps",
        "//src/main/java/com/google/devtools/build/lib/rules/android",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-rules",
        "//src/main/java/com/google/devtools/build/lib/rules/proto",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
    ],
)

java_library(
    name = "build-request-options",
    srcs = ["buildtool/BuildRequestOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:resource_converter",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "runtime/blaze_server_startup_options",
    srcs = ["runtime/BlazeServerStartupOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "keep-going-option",
    srcs = ["runtime/KeepGoingOption.java"],
    deps = ["//src/main/java/com/google/devtools/common/options"],
)

java_library(
    name = "loading-phase-threads-option",
    srcs = ["runtime/LoadingPhaseThreadsOption.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/util:TestType",
        "//src/main/java/com/google/devtools/build/lib/util:resource_converter",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:flogger",
    ],
)

java_library(
    name = "runtime/blaze_command_result",
    srcs = [
        "runtime/BlazeCommandResult.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/util:crash_failure_details",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/protobuf:command_server_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "runtime/build_event_streamer_utils",
    srcs = ["runtime/BuildEventStreamerUtils.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/protobuf:test_status_java_proto",
    ],
)

java_library(
    name = "runtime/command_dispatcher",
    srcs = [
        "runtime/CommandDispatcher.java",
    ],
    deps = [
        ":runtime/blaze_command_result",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/protobuf:invocation_policy_java_proto",
    ],
)

java_library(
    name = "runtime",
    srcs = glob(
        [
            "runtime/*.java",
            "buildtool/*.java",
            "buildtool/buildevent/*.java",
        ],
        exclude = [
            "buildtool/BuildRequestOptions.java",
            "runtime/BlazeCommandResult.java",
            "runtime/CommandDispatcher.java",
            "runtime/KeepGoingOption.java",
            "runtime/LoadingPhaseThreadsOption.java",
        ],
    ),
    deps = [
        ":build-request-options",
        ":keep-going-option",
        ":loading-phase-threads-option",
        ":runtime/blaze_command_result",
        ":runtime/command_dispatcher",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_phase_complete_event",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/analysis:build_info_event",
        "//src/main/java/com/google/devtools/build/lib/analysis:build_view",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/core_option_converters",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/invalid_configuration_exception",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:file_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:no_build_event",
        "//src/main/java/com/google/devtools/build/lib/analysis:no_build_request_finished_event",
        "//src/main/java/com/google/devtools/build/lib/analysis:server_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/coverage_report_action_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis:top_level_artifact_context",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis:view_creation_failed_exception",
        "//src/main/java/com/google/devtools/build/lib/analysis:workspace_status_action",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/cmdline:LabelValidator",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/exec:bin_tools",
        "//src/main/java/com/google/devtools/build/lib/exec:blaze_executor",
        "//src/main/java/com/google/devtools/build/lib/exec:check_up_to_date_filter",
        "//src/main/java/com/google/devtools/build/lib/exec:execution_options",
        "//src/main/java/com/google/devtools/build/lib/exec:executor_builder",
        "//src/main/java/com/google/devtools/build/lib/exec:executor_lifecycle_listener",
        "//src/main/java/com/google/devtools/build/lib/exec:module_action_context_registry",
        "//src/main/java/com/google/devtools/build/lib/exec:remote_local_fallback_registry",
        "//src/main/java/com/google/devtools/build/lib/exec:single_build_file_cache",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_cache",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_strategy_registry",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_strategy_resolver",
        "//src/main/java/com/google/devtools/build/lib/exec:symlink_tree_strategy",
        "//src/main/java/com/google/devtools/build/lib/exec:test_log_helper",
        "//src/main/java/com/google/devtools/build/lib/exec/local:options",
        "//src/main/java/com/google/devtools/build/lib/metrics:event",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:build_type",
        "//src/main/java/com/google/devtools/build/lib/packages:starlark_semantics_options",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/platform:suspend_counter",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:google-auto-profiler-utils",
        "//src/main/java/com/google/devtools/build/lib/profiler/memory:allocationtracker",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/query2:aquery-utils",
        "//src/main/java/com/google/devtools/build/lib/query2/engine",
        "//src/main/java/com/google/devtools/build/lib/query2/query/output",
        "//src/main/java/com/google/devtools/build/lib/rules:alias",
        "//src/main/java/com/google/devtools/build/lib/runtime/events",
        "//src/main/java/com/google/devtools/build/lib/server:rpc_server",
        "//src/main/java/com/google/devtools/build/lib/server/signal",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skyframe:action_execution_inactivity_watchdog",
        "//src/main/java/com/google/devtools/build/lib/skyframe:actiongraph/v2/aquery_output_handler",
        "//src/main/java/com/google/devtools/build/lib/skyframe:aspect_value_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:build_configuration_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:builder",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configuration_phase_started_event",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_progress_receiver",
        "//src/main/java/com/google/devtools/build/lib/skyframe:diff_awareness",
        "//src/main/java/com/google/devtools/build/lib/skyframe:execution_finished_event",
        "//src/main/java/com/google/devtools/build/lib/skyframe:loading_phase_started_event",
        "//src/main/java/com/google/devtools/build/lib/skyframe:managed_directories_knowledge",
        "//src/main/java/com/google/devtools/build/lib/skyframe:package_progress_receiver",
        "//src/main/java/com/google/devtools/build/lib/skyframe:package_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:precomputed_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_value_dirtiness_checker",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_cluster",
        "//src/main/java/com/google/devtools/build/lib/skyframe:target_pattern_phase_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:top_down_action_cache",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:TestType",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:crash_failure_details",
        "//src/main/java/com/google/devtools/build/lib/util:custom_exit_code_publisher",
        "//src/main/java/com/google/devtools/build/lib/util:custom_failure_detail_publisher",
        "//src/main/java/com/google/devtools/build/lib/util:debug-logger-configurator",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:interrupted_failure_details",
        "//src/main/java/com/google/devtools/build/lib/util:logging",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:process",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:output_service",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/windows",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:action_cache_java_proto",
        "//src/main/protobuf:analysis_java_proto",
        "//src/main/protobuf:command_line_java_proto",
        "//src/main/protobuf:command_server_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/main/protobuf:option_filters_java_proto",
        "//src/main/protobuf:test_status_java_proto",
        "//third_party:auto_value",
        "//third_party:error_prone_annotations",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "all-build-rules",
    exports = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:legacy_exports",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/rules:legacy_exports",
        "//src/main/java/com/google/devtools/build/lib/rules/objc",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

########################################################################
#
# The Blaze query language, revision 2: query2.
#

java_library(
    name = "query2",
    exports = [
        ":foundation",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_java_proto",
        "//third_party:guava",
    ],
)

########################################################################
#
# Documentation generation
#

filegroup(
    name = "generated_docs",
    srcs = [
        "//src/main/java/com/google/devtools/build/lib:gen_buildencyclopedia",
        "//src/main/java/com/google/devtools/build/lib:gen_command-line-reference",
        "//src/main/java/com/google/devtools/build/lib:gen_skylarklibrary",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "docs_embedded_in_sources",
    srcs = glob(["**/*.java"]) + [
        "//src/main/java/com/google/devtools/build/lib/analysis:srcs",
        "//src/main/java/com/google/devtools/build/lib/exec:srcs",
        "//src/main/java/com/google/devtools/build/lib/repository:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules:srcs",
    ],
)

genrule(
    name = "gen_buildencyclopedia",
    srcs = [":docs_embedded_in_sources"],
    outs = ["build-encyclopedia.zip"],
    cmd = (
        "mkdir -p $(@D)/be && " +
        "$(location //src/main/java/com/google/devtools/build/docgen:docgen_bin)" +
        " --product_name=bazel" +
        " --provider=com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider" +
        " --input_dir=$$PWD/src/main/java/com/google/devtools/build/lib" +
        " --output_dir=$(@D)/be &&" +
        "zip -qj $@ $(@D)/be/*"
    ),
    tools = [
        "//src/main/java/com/google/devtools/build/docgen:docgen_bin",
        "//src/main/java/com/google/devtools/build/docgen:docgen_javalib",
    ],
    visibility = ["//site:__pkg__"],
)

genrule(
    name = "gen_command-line-reference",
    srcs = [
        "//site:command-line-reference-prefix.html",
        "//site:command-line-reference-suffix.html",
    ],
    outs = ["command-line-reference.html"],
    cmd = (
        "cat $(location //site:command-line-reference-prefix.html) > $@ && " +
        "TMP=`mktemp -d /tmp/tmp.XXXXXXXXXX` && " +
        # TODO(#11179): Remove when the stub template fix is released.
        "export JARBIN=$(JAVABASE)/bin/jar && " +
        "$(location //src/main/java/com/google/devtools/build/lib/bazel:BazelServer) " +
        "--jvm_flag=-Dio.bazel.EnableJni=0 --batch " +
        "--install_base=$${TMP} --output_base=$${TMP}/output/ --output_user_root=$${TMP} " +
        "--failure_detail_out=$${TMP}/output/failure_detail.rawproto " +
        "help everything-as-html >> $@ 2>/dev/null && " +
        "cat $(location //site:command-line-reference-suffix.html) >> $@"
    ),
    toolchains = ["@bazel_tools//tools/jdk:current_host_java_runtime"],
    tools = [
        "//src/main/java/com/google/devtools/build/lib/bazel:BazelServer",
    ],
    visibility = ["//site:__pkg__"],
)

genrule(
    name = "gen_skylarklibrary",
    outs = ["skylark-library.zip"],
    cmd = "mkdir -p $(@D)/skylark-lib &&" +
          "$(location //src/main/java/com/google/devtools/build/docgen:skydoc_bin) $(@D)/skylark-lib" +
          " && zip -qj $@ $(@D)/skylark-lib/*",
    tools = ["//src/main/java/com/google/devtools/build/docgen:skydoc_bin"],
    visibility = ["//site:__pkg__"],
)
