package(
    default_testonly = 1,
    default_visibility = ["//src/tools/remote:__pkg__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/tools/remote:__pkg__"],
)

java_test(
    name = "worker",
    srcs = glob(["*.java"]),
    test_class = "com.google.devtools.build.remote.worker.HttpCacheServerHandlerTest",
    deps = [
        "//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
