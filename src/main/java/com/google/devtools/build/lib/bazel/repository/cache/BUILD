load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

# Description:
#   Main Java code for Bazel
java_library(
    name = "cache",
    srcs = ["RepositoryCache.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/bazel",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "events",
    srcs = ["RepositoryCacheHitEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
    ],
)
