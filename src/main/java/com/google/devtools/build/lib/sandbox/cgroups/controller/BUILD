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
        "//src/main/java/com/google/devtools/build/lib/actions:exec_exception",
        "//src/main/protobuf:failure_details_java_proto",
    ],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups:__subpackages__",
    ],
)
