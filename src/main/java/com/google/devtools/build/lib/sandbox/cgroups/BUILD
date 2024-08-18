load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "cgroups",
    srcs = [
        "Hierarchy.java",
        "Mount.java",
        "VirtualCgroup.java",
        "VirtualCgroupFactory.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox:cgroups_info",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller:v1",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller:v2",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
