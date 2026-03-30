load("@rules_java//java:defs.bzl", "java_test")
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

java_test(
    name = "zstd",
    srcs = glob(
        ["*.java"],
        exclude = [
            "ZstdBenchmark.java",
        ],
    ),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote/zstd",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "@maven//:org_openjdk_jmh_jmh_core",
        "@zstd-jni",
    ],
)

java_opt_binary(
    name = "zstd_benchmark",
    srcs = ["ZstdBenchmark.java"],
    main_class = "org.openjdk.jmh.Main",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote/zstd",
        "//third_party:jmh",
    ],
)
