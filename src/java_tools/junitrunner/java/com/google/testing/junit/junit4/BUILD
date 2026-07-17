# Description:
#   JUnit 4.x extensions

load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//visibility:public"],
)

# Libraries
# =========================================================

# Extensions for writing custom JUnit4 runners
java_library(
    name = "runner",
    testonly = 0,  # TODO(bazel-team): make it testonly
    srcs = glob(["runner/*.java"]),
    deps = [
        "//third_party:junit4",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**/*.java"]) + ["BUILD"],
    visibility = ["//src:__subpackages__"],
)
