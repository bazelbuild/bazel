load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "buildevent",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/buildtool:build_request",
        "//src/main/java/com/google/devtools/build/lib/buildtool:build_result",
        "//src/main/java/com/google/devtools/build/lib/buildtool:execution_progress_receiver",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/metrics/criticalpath",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/runtime:command_line_event",
        "//src/main/java/com/google/devtools/build/lib/runtime:instrumentation_output",
        "//src/main/java/com/google/devtools/build/lib/skyframe/config",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:output_service",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:command_server_java_proto",
        "//third_party/java/auto:auto_value",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/protobuf:protobuf_lite",
        "//third_party/java/protobuf/util:time",
    ],
)
