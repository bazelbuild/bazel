# Description:
#   Tool for desugaring Java constructs not supported by Android tools or devices.

load("@rules_java//java:defs.bzl", "java_binary", "java_library")

filegroup(
    name = "embedded_tools",
    srcs = ["BUILD.tools"],
    visibility = [
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
    ],
)

java_library(
    name = "deps_collector_api",
    srcs = ["DependencyCollector.java"],
    visibility = [
        "//src/test/java/com/google/devtools/build/android/desugar:__subpackages__",
        "//src/tools/android/java/com/google/devtools/build/android:__subpackages__",
        "//src/tools/android/java/com/google/devtools/build/android/r8:__subpackages__",
    ],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/r8:utils",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "desugar",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/test/java/com/google/devtools/build/android/desugar:__pkg__",
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:__pkg__",
    ],
    runtime_deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/dependencies",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/runtime:string_concats",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/runtime:throwable_extension",
    ],
    deps = [
        ":deps_collector_api",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/config",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/corelibadapter",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/covariantreturn",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/nest",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/preanalysis",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/retarget",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/retarget:retarget_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/strconcat",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/typeannotation",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/typehierarchy",
        "//src/tools/android/java/com/google/devtools/build/android/r8:utils",
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_binary(
    name = "Desugar",
    main_class = "com.google.devtools.build.android.desugar.Desugar",
    visibility = [
        "//src/test/java/com/google/devtools/build/android/desugar:__pkg__",
    ],
    runtime_deps = [":desugar"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/dependencies:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/nest:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/covariantreturn:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/retarget:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/typeannotation:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/typehierarchy:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/config:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/corelibadapter:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/runtime:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/strconcat:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/preanalysis:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/scan:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:srcs",
    ],
    visibility = ["//src/tools/android/java/com/google/devtools/build/android:__pkg__"],
)
