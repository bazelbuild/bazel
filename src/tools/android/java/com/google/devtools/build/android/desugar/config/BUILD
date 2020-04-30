load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = [
        "//src/test/java/com/google/devtools/build/android/desugar:__subpackages__",
        "//src/tools/android/java/com/google/devtools/build/android/desugar:__subpackages__",
    ],
)

java_library(
    name = "config",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/android/java/com/google/devtools/build/android/desugar:__pkg__"],
)
