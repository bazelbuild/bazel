load("@rules_java//java:defs.bzl", "java_test")

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
    name = "circuitbreaker",
    srcs = glob(["*.java"]),
    tags = ["manual"],  # Flaky (see b/287004093)
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote",
        "//src/main/java/com/google/devtools/build/lib/remote/circuitbreaker",
        "//src/main/java/com/google/devtools/build/lib/remote/common:cache_not_found_exception",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//third_party:junit4",
        "//third_party:truth",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
