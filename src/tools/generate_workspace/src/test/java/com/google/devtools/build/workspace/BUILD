licenses(["notice"])  # Apache 2.0

java_test(
    name = "WorkspaceTest",
    srcs = ["WorkspaceFileGeneratorTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace/maven",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
