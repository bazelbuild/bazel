load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/concurrent/safeexecutor:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

# Library of concurrency utilities.
java_library(
    name = "concurrent",
    srcs = glob(
        ["*.java"],
        exclude = [
            "SettableFutureKeyedValue.java",
            "TaskDeduplicator.java",
            "ThreadSafety.java",
        ],
    ),
    deps = [
        ":thread_safety",
        "//src/main/java/com/google/devtools/build/lib/concurrent/safeexecutor:safe_executor",
        "//src/main/java/com/google/devtools/build/lib/unsafe:unsafe-provider",
        "//third_party/java/flogger",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/guava:primitives",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "thread_safety",
    srcs = ["ThreadSafety.java"],
)

java_library(
    name = "settable_future_keyed_value",
    srcs = ["SettableFutureKeyedValue.java"],
    deps = [
        "//third_party/java/guava:base",
        "//third_party/java/guava:concurrent",
    ],
)

java_library(
    name = "task_deduplicator",
    srcs = ["TaskDeduplicator.java"],
    deps = [
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:concurrent",
        "//third_party/java/jsr305_annotations",
    ],
)
