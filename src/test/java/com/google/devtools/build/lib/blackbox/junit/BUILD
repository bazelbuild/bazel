java_test(
    name = "TimeoutTestWatcherTest",
    srcs = [
        "TimeoutTestWatcher.java",
        "TimeoutTestWatcherBaseTest.java",
        "TimeoutTestWatcherTest.java",
    ],
    tags = ["black_box_test"],
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox/tests:__pkg__"],
    deps = [
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox:__subpackages__"],
)

java_library(
    name = "test_base",
    testonly = 1,
    srcs = glob(["*.java"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox/tests:__pkg__"],
    deps = [
        "//src/test/java/com/google/devtools/build/lib/blackbox:base_deps",
        "//src/test/java/com/google/devtools/build/lib/blackbox:test_deps",
        "//src/test/java/com/google/devtools/build/lib/blackbox/bazel:common_tools",
        "//src/test/java/com/google/devtools/build/lib/blackbox/bazel:context",
        "//src/test/java/com/google/devtools/build/lib/blackbox/framework",
    ],
)
