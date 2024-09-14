# Description:
#   Tools for scanning bytecode for references to other classes and emitting Proguard-style keeps.

load("@rules_java//java:defs.bzl", "java_binary", "java_library")

filegroup(
    name = "embedded_tools",
    srcs = ["BUILD.tools"],
    visibility = [
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
    ],
)

java_library(
    name = "scan",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
    ],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//src/tools/android/java/com/google/devtools/build/android:android_options_utils",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "@maven//:com_beust_jcommander",
    ],
)

java_binary(
    name = "KeepScanner",
    testonly = 1,  # normally used through @bazel_tools
    main_class = "com.google.devtools.build.android.desugar.scan.KeepScanner",
    visibility = [
        "//src/test/java/com/google/devtools/build/android/desugar/scan:__pkg__",
    ],
    runtime_deps = [":scan"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/android/java/com/google/devtools/build/android/desugar:__pkg__"],
)
