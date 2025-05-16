load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//src/java_tools/junitrunner:junitrunner_package_group",
    ],
)

# TODO(bazel-team): This should be testonly = 1.
java_library(
    name = "sharding",
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/api",
        "//third_party:junit4",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + [
        "BUILD",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/api:srcs",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/testing:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)
