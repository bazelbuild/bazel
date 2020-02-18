load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/unix/jni:srcs",
    ],
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "unix",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/unix/jni",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:flogger",
        "//third_party:guava",
    ],
)
