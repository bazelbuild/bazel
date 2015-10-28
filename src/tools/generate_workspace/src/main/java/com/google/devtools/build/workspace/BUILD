java_library(
    name = "workspace",
    srcs = glob(["*.java"]),
    visibility = ["//src/tools/generate_workspace:__pkg__"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:analysis-exec-rules-skyframe",
        "//src/main/java/com/google/devtools/build/lib:bazel-core",
        "//src/main/java/com/google/devtools/build/lib:buildtool-runtime",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace/maven",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace/maven:rule",
        "//third_party:guava",
        "//third_party:maven_model",
    ],
)
