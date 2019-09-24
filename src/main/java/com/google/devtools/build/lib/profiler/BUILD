load("@rules_java//java:defs.bzl", "java_binary", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "profiler",
    srcs = [
        "AutoProfiler.java",
        "CollectLocalCpuUsage.java",
        "MemoryProfiler.java",
        "MetricData.java",
        "PredicateBasedStatRecorder.java",
        "ProfilePhase.java",
        "ProfilePhaseStatistics.java",
        "Profiler.java",
        "ProfilerTask.java",
        "SilentCloseable.java",
        "SingleStatRecorder.java",
        "StatRecorder.java",
        "TimeSeries.java",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:base-util",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "google-auto-profiler-utils",
    srcs = ["GoogleAutoProfilerUtils.java"],
    deps = [
        ":profiler",
        "//third_party:flogger",
    ],
)

# Profiler chart library.
java_library(
    name = "profiler-output",
    srcs = glob([
        "analysis/*.java",
        "chart/*.java",
        "output/*.java",
        "statistics/*.java",
    ]),
    deps = [
        ":profiler",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_binary(
    name = "profiler-grapher",
    srcs = glob([
        "grapher/*.java",
    ]),
    main_class = "com.google.devtools.build.lib.profiler.grapher.ProfileGrapher",
    deps = [
        ":profiler",
        "//third_party:gson",
        "//third_party:guava",
    ],
)
