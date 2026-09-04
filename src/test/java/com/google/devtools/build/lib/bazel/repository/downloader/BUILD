load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/bazel/repository:__pkg__"],
)

java_library(
    name = "DownloaderTestSuite_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/cache",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/events:event_bus_event_handler",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/net/starlark/java/syntax",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party/java/auth_kit",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:hash",
        "//third_party/java/guava:io",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "DownloaderTestSuite",
    size = "medium",
    flaky = True,
    tags = [
        "requires-network",
        "rules",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":DownloaderTestSuite_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)
