load("@rules_java//java:defs.bzl", "java_library")

package(default_applicable_licenses = ["//:license"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller/v1:srcs",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller/v2:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "controller",
    srcs = [
        "Controller.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
    ],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib/metrics:__subpackages__",
    ],
)
