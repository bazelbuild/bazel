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
    name = "collect_local_resource_usage",
    srcs = [
        "CollectLocalResourceUsage.java",
    ],
    deps = [
        ":network_metrics_collector",
        ":profiler",
        ":system_network_stats_service",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/unix:procmeminfo_parser",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:resource_usage",
        "//src/main/java/com/google/devtools/build/lib/worker:worker_process_metrics",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# Profiler interface and helper classes. Available in both the SC and the LC.
java_library(
    name = "profiler",
    srcs = [
        "AsyncProfiler.java",
        "AutoProfiler.java",
        "CounterSeriesCollector.java",
        "CounterSeriesTask.java",
        "LocalResourceCollector.java",
        "MemoryProfiler.java",
        "NoOpTraceProfilerService.java",
        "ProfilePhase.java",
        "Profiler.java",
        "ProfilerTask.java",
        "SilentCloseable.java",
        "SlowTask.java",
        "StatRecorder.java",
        "TimeSeries.java",
        "TraceProfilerService.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime/blaze_service",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/util:TestType",
        "//src/main/java/com/google/devtools/build/lib/util:heap_offset_helper",
        "//src/main/java/com/google/devtools/build/lib/util:pair",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auto_value",
        "//third_party:error_prone_annotations",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# SC only
java_library(
    name = "profiler_impl",
    srcs = [
        "CounterSeriesTraceData.java",
        "JsonTraceFileWriter.java",
        "MetricData.java",
        "PredicateBasedStatRecorder.java",
        "SingleStatRecorder.java",
        "TaskData.java",
        "ThreadMetadata.java",
        "TimeSeriesImpl.java",
        "TraceData.java",
        "TraceProfilerServiceImpl.java",
    ],
    deps = [
        ":profiler",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/collect:extrema",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//third_party:auto_value",
        "//third_party:error_prone_annotations",
        "//third_party:flogger",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "command_profiler_module",
    srcs = ["CommandProfilerModule.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:async_profiler",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "network_metrics_collector",
    srcs = ["NetworkMetricsCollector.java"],
    deps = [
        ":system_network_stats_service",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "system_network_stats_service",
    srcs = ["SystemNetworkStatsService.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib:runtime/blaze_service"],
)

java_library(
    name = "system_network_stats_service_impl",
    srcs = ["SystemNetworkStatsServiceImpl.java"],
    deps = [
        ":system_network_stats_service",
        "//src/main/java/com/google/devtools/build/lib/jni",
        "//src/main/java/com/google/devtools/build/lib/util:os",
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
    testonly = 1,
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
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
