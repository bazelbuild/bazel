java_binary(
    name = "generate_workspace",
    data = ["//src:libunix"],
    jvm_flags = ["-Djava.library.path=$(GENDIR)/src/main/native"],
    main_class = "com.google.devtools.build.workspace.GenerateWorkspace",
    visibility = ["//visibility:public"],
    runtime_deps = [":workspace"],
)

java_library(
    name = "workspace",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java:analysis-exec-rules-skyframe",
        "//src/main/java:bazel-core",
        "//src/main/java:events",
        "//src/main/java:options",
        "//src/main/java:packages",
        "//src/main/java:vfs",
        "//src/main/java/com/google/devtools/build/workspace/maven",
        "//src/main/java/com/google/devtools/build/workspace/maven:rule",
        "//third_party:guava",
    ],
)
