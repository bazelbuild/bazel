java_library(
    name = "workspace",
    srcs = glob(["*.java"]),
    visibility = ["//src/tools/generate_workspace:__subpackages__"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace/maven",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace/maven:rule",
        "//third_party:guava",
        "//third_party:maven_model",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace/maven:srcs"],
    visibility = ["//src/tools/generate_workspace:__pkg__"],
)
