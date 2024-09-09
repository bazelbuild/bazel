package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "http",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/test/java/com/google/devtools/build/lib/remote/http:__pkg__",
        "//src/tools/remote:__subpackages__",
    ],
    deps = [
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:netty",
    ],
)
