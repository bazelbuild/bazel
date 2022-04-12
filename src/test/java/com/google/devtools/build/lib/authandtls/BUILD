load("@rules_java//java:defs.bzl", "java_library", "java_test")

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

java_library(
    name = "AuthAndTlsTests_lib",
    srcs = glob(
        [
            "*.java",
        ],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "AuthAndTlsTests",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":AuthAndTlsTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)
