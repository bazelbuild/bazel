load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller:srcs",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/proto:srcs",
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
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/proto:cgroups_info_java_proto",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//third_party/java/auto:auto_value",
        "//third_party/java/flogger",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:io",
        "//third_party/java/jsr305_annotations",
    ],
)
