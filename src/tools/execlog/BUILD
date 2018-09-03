filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/execlog/src/main/java/com/google/devtools/build/execlog:srcs",
        "//src/tools/execlog/test/main/java/com/google/devtools/build/execlog:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

java_binary(
    name = "parser",
    main_class = "com.google.devtools.build.execlog.ExecLogParser",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/tools/execlog/src/main/java/com/google/devtools/build/execlog:parser"],
)
