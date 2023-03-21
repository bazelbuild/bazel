load("@rules_java//java:defs.bzl", "java_binary")

package(default_visibility = ["//:__pkg__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker:srcs",
        "//src/tools/remote/src/test/java/com/google/devtools/build/remote/worker:srcs",
        "//src/tools/remote/src/test/java/com/google/devtools/build/remote/worker/http:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_binary(
    name = "worker",
    main_class = "com.google.devtools.build.remote.worker.RemoteWorker",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker"],
)
