load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//src/java_tools/junitrunner:junitrunner_package_group",
    ],
)

# TODO(bazel-team): This should be testonly = 1.
java_library(
    name = "api",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party/java/junit:not-testonly",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + ["BUILD"],
)
