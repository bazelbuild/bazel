java_library(
    name = "workspace",
    srcs = glob(["*.java"]),
    visibility = ["//src/tools/generate_workspace:__pkg__"],
    deps = [
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:buildtool-runtime",
        "//src/main/java:events",
        "//src/main/java:os_util",
        "//src/main/java:packages",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace/maven",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace/maven:rule",
        "//third_party:guava",
        "//third_party:maven_model",
    ],
)
