load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

# Unit test for BuildFileContentsGenerator
java_test(
    name = "BuildFileContentsGeneratorTest",
    size = "small",
    srcs = ["BuildFileContentsGeneratorTest.java"],
    runtime_deps = ["//src/main/java/com/google/devtools/build/lib/bazel/rules"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/net/starlark/java/syntax",
        "//src/test/java/com/google/devtools/build/lib/generatedprojecttest/util:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
