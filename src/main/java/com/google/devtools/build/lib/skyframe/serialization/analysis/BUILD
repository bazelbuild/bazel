load("@rules_java//java:java_library.bzl", "java_library")

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
    name = "event_listener",
    srcs = ["RemoteAnalysisCachingEventListener.java"],
    deps = [
        ":client_id",
        ":remote_analysis_cache_client",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:frontier_node_version",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "dependencies_provider",
    srcs = [
        "RemoteAnalysisCacheReaderDepsProvider.java",
        "RemoteAnalysisCachingDependenciesProvider.java",
    ],
    deps = [
        ":options",
        ":remote_analysis_cache_client",
        ":remote_analysis_caching_server_state",
        ":remote_analysis_json_log_writer",
        ":remote_analysis_metadata_writer",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:frontier_node_version",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "frontier_serializer",
    srcs = [
        "FrontierSerializer.java",
        "SelectedEntrySerializer.java",
    ],
    deps = [
        ":dependencies_provider",
        ":event_listener",
        ":file_dependency_serializer",
        ":file_op_node_map",
        ":invalidation_data_reference_or_future",
        ":long_version_getter_test_injection",
        ":missing_skyframe_entry_exception",
        ":options",
        ":remote_analysis_json_log_writer",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_data",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_key",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target_value",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/rules:alias",
        "//src/main/java/com/google/devtools/build/lib/skyframe:action_execution_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:action_template_expansion_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:filesystem_keys",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:error_message_helper",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:frontier_node_version",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:write_statuses",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:registered_execution_platforms_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:registered_toolchains_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:toolchain_context_key",
        "//src/main/java/com/google/devtools/build/lib/util:TestType",
        "//src/main/java/com/google/devtools/build/lib/versioning:long_version_getter",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/main/protobuf:file_invalidation_data_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "file_op_node_map",
    srcs = ["FileOpNodeMemoizingLookup.java"],
    deps = [
        ":missing_skyframe_entry_exception",
        ":value_or_future_map",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_key",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe:filesystem_keys",
        "//src/main/java/com/google/devtools/build/lib/skyframe:rule_configured_target_value",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "file_dependency_serializer",
    srcs = ["FileDependencySerializer.java"],
    deps = [
        ":file_dependency_key_support",
        ":invalidation_data_reference_or_future",
        ":magic_bytes",
        ":missing_skyframe_entry_exception",
        ":value_or_future_map",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe:filesystem_keys",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:write_statuses",
        "//src/main/java/com/google/devtools/build/lib/util:TestType",
        "//src/main/java/com/google/devtools/build/lib/versioning:long_version_getter",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/protobuf:file_invalidation_data_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "@com_google_protobuf//:protobuf_java",
        "@zstd-jni",
    ],
)

java_library(
    name = "invalidation_data_reference_or_future",
    srcs = ["InvalidationDataInfoOrFuture.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent:settable_future_keyed_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:filesystem_keys",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:write_statuses",
        "//src/main/java/com/google/devtools/build/lib/vfs",
    ],
)

java_library(
    name = "file_dependency_deserializer",
    srcs = [
        "FileDependencies.java",
        "FileDependencyDeserializer.java",
        "FileSystemDependencies.java",
        "ListingDependencies.java",
        "NestedDependencies.java",
    ],
    deps = [
        ":file_dependency_key_support",
        ":magic_bytes",
        ":value_or_future_map",
        ":versioned_changes",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:settable_future_keyed_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/versioning:long_version_getter",
        "//src/main/java/com/google/devtools/build/lib/vfs:ospathpolicy",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/protobuf:file_invalidation_data_java_proto",
        "//third_party:caffeine",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
        "@com_google_protobuf//:protobuf_java",
        "@zstd-jni",
    ],
)

java_library(
    name = "versioned_changes_validator",
    srcs = [
        "AlwaysMatch.java",
        "FileOpMatchMemoizingLookup.java",
        "FileOpMatchResultTypes.java",
        "MatchIndicator.java",
        "NestedMatchMemoizingLookup.java",
        "NestedMatchResultTypes.java",
        "NoMatch.java",
        "VersionedChangesValidator.java",
    ],
    deps = [
        ":file_dependency_deserializer",
        ":value_or_future_map",
        ":versioned_changes",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:settable_future_keyed_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "file_dependency_key_support",
    srcs = ["FileDependencyKeySupport.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/versioning:long_version_getter",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/protobuf:file_invalidation_data_java_proto",
        "//third_party:guava",
    ],
)

java_library(
    name = "options",
    srcs = ["RemoteAnalysisCachingOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "value_or_future_map",
    srcs = [
        "AbstractValueOrFutureMap.java",
        "ValueOrFutureMap.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent:settable_future_keyed_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "client_id",
    srcs = ["ClientId.java"],
    deps = [
        "//third_party:error_prone_annotations",
        # Do NOT add Bazel-specific dependencies here.
    ],
)

java_library(
    name = "versioned_changes",
    srcs = ["VersionedChanges.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "magic_bytes",
    srcs = ["MagicBytes.java"],
    deps = ["//third_party:guava"],
)

java_library(
    name = "remote_analysis_cache_client",
    srcs = ["RemoteAnalysisCacheClient.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis/proto:analysis_cache_service_metadata_status_java_proto",
        "//src/main/java/com/google/devtools/build/lib/util:decimal_bucketer",
        "//third_party:guava",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "remote_analysis_caching_services_supplier",
    srcs = ["RemoteAnalysisCachingServicesSupplier.java"],
    deps = [
        ":client_id",
        ":options",
        ":remote_analysis_cache_client",
        ":remote_analysis_json_log_writer",
        ":remote_analysis_metadata_writer",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "long_version_getter_test_injection",
    srcs = ["LongVersionGetterTestInjection.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:TestType",
        "//src/main/java/com/google/devtools/build/lib/versioning:long_version_getter",
        "//third_party:guava",
    ],
)

java_library(
    name = "analysis_cache_invalidator",
    srcs = ["AnalysisCacheInvalidator.java"],
    deps = [
        ":client_id",
        ":remote_analysis_cache_client",
        ":remote_analysis_caching_server_state",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:frontier_node_version",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:flogger",
        "//third_party:guava",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "remote_analysis_caching_server_state",
    srcs = ["RemoteAnalysisCachingServerState.java"],
    deps = [
        ":client_id",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:frontier_node_version",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "remote_analysis_metadata_writer",
    srcs = ["RemoteAnalysisMetadataWriter.java"],
)

java_library(
    name = "missing_skyframe_entry_exception",
    srcs = ["MissingSkyframeEntryException.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
    ],
)

java_library(
    name = "remote_analysis_json_log_writer",
    srcs = ["RemoteAnalysisJsonLogWriter.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:write_statuses",
        "//third_party:flogger",
        "//third_party:gson",
        "//third_party:guava",
    ],
)

java_library(
    name = "remote_analysis_cache_manager",
    srcs = ["RemoteAnalysisCacheManager.java"],
    deps = [
        ":analysis_cache_invalidator",
        ":client_id",
        ":dependencies_provider",
        ":event_listener",
        ":frontier_serializer",
        ":options",
        ":remote_analysis_cache_client",
        ":remote_analysis_caching_server_state",
        ":remote_analysis_caching_services_supplier",
        ":remote_analysis_json_log_writer",
        ":remote_analysis_metadata_writer",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/test_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:invalid_configuration_exception",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect:path_fragment_prefix_trie",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache:package_path_codec_dependencies",
        "//src/main/java/com/google/devtools/build/lib/runtime",
        "//src/main/java/com/google/devtools/build/lib/runtime:instrumentation_output",
        "//src/main/java/com/google/devtools/build/lib/skyframe:prerequisite_package_function",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyfocus_options",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_cluster",
        "//src/main/java/com/google/devtools/build/lib/skyframe:workspace_info",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:frontier_node_version",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:flogger",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
