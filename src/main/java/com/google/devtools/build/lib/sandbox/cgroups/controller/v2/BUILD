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
        "UnifiedController.java",
        "UnifiedCpu.java",
        "UnifiedMemory.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller",
        "//third_party:guava",
    ],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/sandbox/cgroups:__pkg__",
    ],
)
