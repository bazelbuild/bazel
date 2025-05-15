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
    name = "GraphTests_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "GraphTests",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":GraphTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib/testutil",
    ],
)
