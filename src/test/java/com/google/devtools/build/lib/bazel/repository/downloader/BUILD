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
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "DownloaderTestSuite_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/cache",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/net/starlark/java/syntax",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:auth",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "DownloaderTestSuite",
    size = "medium",
    flaky = True,
    jvm_flags = [
        "-Djava.lang.Thread.allowVirtualThreads=true",
    ],
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
