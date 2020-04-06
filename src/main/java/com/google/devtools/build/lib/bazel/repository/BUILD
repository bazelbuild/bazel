load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]) + [
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/cache:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/skylark:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "repository",
    srcs = glob(["*.java"]),
    resources = [
        "local_config_platform.WORKSPACE",
    ],
    deps = [
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/cache:events",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:apache_commons_compress",
        "//third_party:auto_value",
        "//third_party:diffutils",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:xz",
    ],
)
