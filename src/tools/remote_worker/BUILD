filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/tools/remote_worker/src/main/java/com/google/devtools/build/remote:srcs"],
    visibility = ["//src:__pkg__"],
)

java_binary(
    name = "remote_worker",
    jvm_flags = [
        # Enables REST for Hazelcast server for testing.
        "-Dhazelcast.rest.enabled=true",
        # Set this to the same function that you run Bazel with.
        "-Dbazel.DigestFunction=SHA1",
    ],
    main_class = "com.google.devtools.build.remote.RemoteWorker",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/tools/remote_worker/src/main/java/com/google/devtools/build/remote"],
)
