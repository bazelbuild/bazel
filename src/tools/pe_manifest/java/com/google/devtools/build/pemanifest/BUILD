load("@rules_java//java:defs.bzl", "java_library")

package(default_applicable_licenses = ["//:license"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/pe_manifest:__pkg__"],
)

java_library(
    name = "pemanifest",
    srcs = ["PeManifest.java"],
    visibility = [
        "//src/tools/pe_manifest:__subpackages__",
    ],
)
