java_binary(
    name = "generate_workspace",
    data = ["//src:libunix"],
    jvm_flags = ["-Djava.library.path=$(GENDIR)/src/main/native"],
    main_class = "com.google.devtools.build.workspace.GenerateWorkspace",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace:srcs"],
    visibility = ["//src:__pkg__"],
)
