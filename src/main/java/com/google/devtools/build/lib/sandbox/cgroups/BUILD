load("@rules_java//java:defs.bzl", "java_library")

package(default_applicable_licenses = ["//:license"])

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
        "VirtualCGroup.java",
        "VirtualCGroupFactory.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox:cgroups_info",
        "//src/main/java/com/google/devtools/build/lib/actions:exec_exception",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller/v1",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller/v2",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:guava",
    ],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/metrics:__subpackages__",
        "//src/main/java/com/google/devtools/build/lib/sandbox:__subpackages__",
        "//src/main/java/com/google/devtools/build/lib/worker:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib/metrics:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib/sandbox/cgroups:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib/worker:__subpackages__",
    ],
)
