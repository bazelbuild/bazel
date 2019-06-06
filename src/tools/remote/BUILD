filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker:srcs",
        "//src/tools/remote/src/test/java/com/google/devtools/build/remote/worker:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

java_binary(
    name = "worker",
    main_class = "com.google.devtools.build.remote.worker.RemoteWorker",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker"],
)
