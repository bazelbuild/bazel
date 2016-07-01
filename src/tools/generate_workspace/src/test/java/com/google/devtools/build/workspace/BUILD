licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/generate_workspace/src/test/java/com/google/devtools/build/workspace/maven:srcs",
    ],
    visibility = ["//src/tools/generate_workspace:__pkg__"],
)

java_test(
    name = "ResultWriterTest",
    srcs = ["ResultWriterTest.java"],
    deps = [
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace/maven:rule",
        "//third_party:aether",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
