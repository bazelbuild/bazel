load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "v2",
    srcs = [
        "UnifiedCpu.java",
        "UnifiedMemory.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller",
    ],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups:__pkg__",
    ],
)
