load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "event",
    srcs = ["BuildMetricsEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//third_party:guava",
    ],
)

java_library(
    name = "metrics_module",
    srcs = glob(
        ["*.java"],
        exclude = [
            "BuildMetricsEvent.java",
            "PostGCMemoryUseRecorder.java",
        ],
    ),
    deps = [
        ":event",
        ":memory-use-recorder",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:analysis_graph_stats_event",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_phase_complete_event",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_phase_started_event",
        "//src/main/java/com/google/devtools/build/lib/analysis:no_build_request_finished_event",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:network_metrics_collector",
        "//src/main/java/com/google/devtools/build/lib/skyframe:execution_finished_event",
        "//src/main/java/com/google/devtools/build/lib/skyframe:top_level_status_events",
        "//src/main/java/com/google/devtools/build/lib/worker:worker_metric",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe_graph_stats_event",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
    ],
)

java_library(
    name = "memory-use-recorder",
    srcs = ["PostGCMemoryUseRecorder.java"],
    deps = [
        ":garage-collection-metrics-util",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "garage-collection-metrics-util",
    srcs = ["GarbageCollectionMetricsUtils.java"],
)
