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
    name = "HashTests_lib",
    srcs = glob(
        [
            "*.java",
        ],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/hash",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "HashTests",
    size = "medium",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":HashTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)
