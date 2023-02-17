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
    name = "profiler",
    srcs = [
        "AutoProfiler.java",
        "CollectLocalResourceUsage.java",
        "CounterSeriesTraceData.java",
        "JsonTraceFileWriter.java",
        "MemoryProfiler.java",
        "MetricData.java",
        "PredicateBasedStatRecorder.java",
        "ProfilePhase.java",
        "Profiler.java",
        "ProfilerTask.java",
        "SilentCloseable.java",
        "SingleStatRecorder.java",
        "StatRecorder.java",
        "ThreadMetadata.java",
        "TimeSeries.java",
        "TraceData.java",
    ],
    deps = [
        ":network_metrics_collector",
        "//src/main/java/com/google/devtools/build/lib/actions:resource_estimator",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/collect:extrema",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/unix:procmeminfo_parser",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:pair",
        "//src/main/java/com/google/devtools/build/lib/util:resource_usage",
        "//src/main/java/com/google/devtools/build/lib/worker:worker_metric",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auto_value",
        "//third_party:error_prone_annotations",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "network_metrics_collector",
    srcs = ["NetworkMetricsCollector.java"],
    deps = [
        ":system_network_stats",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "system_network_stats",
    srcs = ["SystemNetworkStats.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/jni",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "google-auto-profiler-utils",
    srcs = ["GoogleAutoProfilerUtils.java"],
    deps = [
        ":profiler",
        "//third_party:flogger",
        "//third_party:guava",
    ],
)

java_library(
    name = "profiler-output",
    srcs =
        [
            "JsonProfile.java",
            "TraceEvent.java",
        ] + glob([
            "output/*.java",
            "statistics/*.java",
        ]),
    deps = [
        ":profiler",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:auto_value",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
