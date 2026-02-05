load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]) + [
        "//src/main/java/net/starlark/java/lib/json:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "lib",
    srcs = glob(["*.java"]),
    visibility = ["//src/main/java/net/starlark/java:clients"],
    deps = [
        "//src/main/java/net/starlark/java/eval",
    ],
)
