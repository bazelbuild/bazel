filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker:__pkg__"],
)

java_library(
    name = "http",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/test/java/com/google/devtools/build/lib/remote/http:__pkg__",
        "//src/tools/remote:__subpackages__",
    ],
    deps = [
        "//third_party:guava",
        "//third_party:netty",
    ],
)
