filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker:srcs"],
    visibility = ["//src:__pkg__"],
)

java_binary(
    name = "worker",
    jvm_flags = [
        # Enables REST for Hazelcast server for testing.
        "-Dhazelcast.rest.enabled=true",
    ],
    main_class = "com.google.devtools.build.remote.worker.RemoteWorker",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker"],
)
