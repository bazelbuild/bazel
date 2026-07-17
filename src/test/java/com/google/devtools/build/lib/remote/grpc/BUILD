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
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:rxjava3",
        "//third_party:truth",
        "//third_party/grpc-java:grpc-jar",
    ],
)
