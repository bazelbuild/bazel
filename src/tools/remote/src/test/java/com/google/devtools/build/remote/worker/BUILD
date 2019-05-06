package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 1,
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
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
