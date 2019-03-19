package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/remote:__pkg__"],
)

java_test(
    name = "http",
    srcs = glob(["*.java"]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote/blobstore/http",
        "//src/main/java/com/google/devtools/build/lib/remote/util",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//third_party:api_client",
        "//third_party:auth",
        "//third_party:guava",
        "//third_party:mockito2",
        "//third_party:netty",
        "//third_party:truth",
    ],
)
