load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "codec-scanning-constants",
    srcs = ["CodecScanningConstants.java"],
)

java_library(
    name = "serialization_registry_setup_helpers",
    srcs = ["SerializationRegistrySetupHelpers.java"],
    deps = [
        ":module_codec",
        ":serialization",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/actions:runfiles_metadata",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/test_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_provider_map_impl",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:provider",
        "//src/main/java/com/google/devtools/build/lib/rules:alias",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/skyframe:configured_target_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:rule_configured_target_value",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
    ],
)

java_library(
    name = "serialization_module",
    srcs = ["SerializationModule.java"],
    deps = [
        ":serialization",
        ":serialization_registry_setup_helpers",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:remote_analysis_cache_client",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:remote_analysis_caching_services_supplier",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "depend_on_future_shim",
    srcs = ["DependOnFutureShim.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "frontier_node_version",
    srcs = ["FrontierNodeVersion.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:client_id",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)

java_library(
    name = "serialization",
    srcs = glob(
        ["**/*.java"],
        exclude = [
            "AbstractExportedStarlarkSymbolCodec.java",
            "CodecScanningConstants.java",
            "DependOnFutureShim.java",
            "DeserializedSkyValue.java",
            "ErrorMessageHelper.java",
            "FrontierNodeVersion.java",
            "ModuleCodec.java",
            "SerializationRegistrySetupHelpers.java",
            "SerializationConstants.java",
            "SerializationModule.java",
            "VisibleForSerialization.java",
            "WriteStatuses.java",
        ],
    ),
    add_opens = [
        "java.base/java.lang.invoke",
    ],
    deps = [
        ":codec-scanning-constants",
        ":constants",
        ":depend_on_future_shim",
        ":frontier_node_version",
        ":write_statuses",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:remote_analysis_cache_client",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/analysis:remote_analysis_json_log_writer",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/unsafe:unsafe-provider",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/protobuf:file_invalidation_data_java_proto",
        "//third_party:caffeine",
        "//third_party:error_prone_annotations",
        "//third_party:fastutil",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/pprof:profile_java_proto",
        "@com_google_protobuf//:protobuf_java",
        "@zstd-jni",
    ],
)

java_library(
    name = "constants",
    srcs = ["SerializationConstants.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:TestType",
    ],
)

java_library(
    name = "visible-for-serialization",
    srcs = ["VisibleForSerialization.java"],
)

java_library(
    name = "deserialized-sky-value",
    srcs = ["DeserializedSkyValue.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
    ],
)

java_library(
    name = "error_message_helper",
    srcs = ["ErrorMessageHelper.java"],
    deps = ["//third_party:guava"],
)

java_library(
    name = "abstract-exported-starlark-symbol-codec",
    srcs = ["AbstractExportedStarlarkSymbolCodec.java"],
    deps = [
        ":serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "module_codec",
    srcs = ["ModuleCodec.java"],
    deps = [
        ":serialization",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "write_statuses",
    srcs = ["WriteStatuses.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
    ],
)
