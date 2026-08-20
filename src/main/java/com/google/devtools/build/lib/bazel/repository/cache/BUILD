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
    srcs = [
        "DownloadCache.java",
        "LocalRepoContentsCache.java",
        "RepositoryCache.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/server:idle_task",
        "//src/main/java/com/google/devtools/build/lib/util:file_system_lock",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:hash",
        "//third_party/java/guava:io",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "events",
    srcs = ["DownloadCacheHitEvent.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib/events"],
)
