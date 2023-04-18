load("@rules_java//java:defs.bzl", "java_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "zstd",
    srcs = glob(["*.java"]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote/zstd",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "@zstd-jni//:zstd-jni",
    ],
)
