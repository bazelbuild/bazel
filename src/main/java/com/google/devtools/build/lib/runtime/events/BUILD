load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "events",
    srcs = glob(["*.java"]),
    deps = ["//src/main/java/com/google/devtools/build/lib/events"],
)
