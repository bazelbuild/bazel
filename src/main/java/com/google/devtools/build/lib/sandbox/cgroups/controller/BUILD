load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "**",
        "v1/**",
        "v2/**",
    ]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "controller",
    srcs = [
        "Controller.java",
    ],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib/metrics:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib/sandbox/cgroups:__subpackages__",
    ],
)

java_library(
    name = "v2",
    srcs = glob(["v2/**"]),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/metrics:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/sandbox/cgroups:__pkg__",
    ],
    deps = [
        ":controller",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:io",
    ],
)

java_library(
    name = "v1",
    srcs = glob(["v1/**"]),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/sandbox/cgroups:__pkg__",
    ],
    deps = [
        ":controller",
    ],
)
