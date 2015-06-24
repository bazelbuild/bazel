java_binary(
    name = "generate_workspace",
    srcs = [
        "GenerateWorkspace.java",
        "GenerateWorkspaceOptions.java",
    ],
    main_class = "com.google.devtools.build.workspace.GenerateWorkspace",
    visibility = ["//visibility:public"],
    deps = [
        ":workspace",
        "//src/main/java:bazel-core",
        "//src/main/java:common",
        "//src/main/java:events",
        "//src/main/java:options",
        "//src/main/java:packages",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/workspace/maven",
    ],
)

java_library(
    name = "workspace",
    srcs = ["Resolver.java"],
    deps = [
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:events",
        "//src/main/java:packages",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/workspace/maven",
        "//src/main/java/com/google/devtools/build/workspace/maven:rule",
        "//third_party:guava",
    ],
)
