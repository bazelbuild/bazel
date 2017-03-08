filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/tools/remote_worker/src/main/java/com/google/devtools/build/remote:srcs"],
    visibility = ["//src:__pkg__"],
)

java_binary(
    name = "remote_worker",
    main_class = "com.google.devtools.build.remote.RemoteWorker",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/tools/remote_worker/src/main/java/com/google/devtools/build/remote"],
)
