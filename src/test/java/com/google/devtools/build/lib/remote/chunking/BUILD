load("@rules_java//java:defs.bzl", "java_library", "java_test")
load("//src:java_opt_binary.bzl", "java_opt_binary")

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
    name = "ChunkingTests_lib",
    srcs = [
        "ChunkingConfigTest.java",
        "FastCDCChunkerTest.java",
    ],
    data = [
        # Public domain image by Toriyama Sekien (1712-1788),
        # first published in "Gazu Hyakki Yagyo" (1776).
        # Source: https://commons.wikimedia.org/wiki/File:SekienAkashita.jpg
        "SekienAkashita.jpg",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote/chunking",
        "//src/main/java/com/google/devtools/build/lib/remote/util:digest_utils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "@rules_java//java/runfiles",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_test(
    name = "ChunkingTests",
    test_class = "com.google.devtools.build.lib.AllTests",
    data = [
        "SekienAkashita.jpg",
    ],
    env = {
        "SEKIEN_AKASHITA_PATH": "$(rlocationpath SekienAkashita.jpg)",
    },
    runtime_deps = [
        ":ChunkingTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

java_opt_binary(
    name = "FastCDCBenchmark",
    srcs = ["FastCDCBenchmark.java"],
    main_class = "org.openjdk.jmh.Main",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote/chunking",
        "//src/main/java/com/google/devtools/build/lib/remote/util:digest_utils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/bazel",
        "//third_party:jmh",
    ],
)
