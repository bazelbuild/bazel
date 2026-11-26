load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

java_test(
    name = "CompressionServiceImplTest",
    size = "small",
    srcs = ["CompressionServiceImplTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/compress:compression_service",
        "//src/main/java/com/google/devtools/build/lib/compress:compression_service_impl",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)
