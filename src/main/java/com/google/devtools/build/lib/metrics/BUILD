load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
)

EVENT_SRCS = ["BuildMetricsEvent.java"]

java_library(
    name = "event",
    srcs = EVENT_SRCS,
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
        exclude = EVENT_SRCS,
    ),
    exports = [
        ":event",
    ],
    deps = [
        ":event",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)
