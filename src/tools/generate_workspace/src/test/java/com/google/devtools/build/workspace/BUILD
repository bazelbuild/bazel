licenses(["notice"])  # Apache 2.0

java_test(
    name = "WorkspaceFileGeneratorTest",
    srcs = ["WorkspaceFileGeneratorTest.java"],
    # This test connects to the network on OS X (the sandbox prevents it from doing so on Linux),
    # so disable it for now.
    tags = ["manual"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace/maven",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
