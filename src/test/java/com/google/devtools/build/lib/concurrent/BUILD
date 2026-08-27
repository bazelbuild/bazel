load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/lib/concurrent/safeexecutor:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "ConcurrentTests_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:settable_future_keyed_value",
        "//src/main/java/com/google/devtools/build/lib/concurrent:task_deduplicator",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/concurrent/safeexecutor:safe_executor",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestThread",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:ThrowingRecordableRunnableWrapper",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "ConcurrentTests",
    size = "medium",
    jvm_flags = [
        # Required for Mockito.spy.
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
    ],
    tags = [
        "foundations",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":ConcurrentTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)
