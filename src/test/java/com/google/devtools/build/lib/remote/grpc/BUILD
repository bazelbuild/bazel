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
)

java_test(
    name = "grpc",
    srcs = glob(["*.java"]),
    tags = [
        "requires-network",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote/grpc",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib/remote/util",
        "//third_party/java/grpc:core",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/rxjava3",
        "//third_party/java/truth",
    ],
)
