package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "util",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:jsr305",
        "//third_party:junit4",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,  # All srcs should be not test only, overwrite package default.
    srcs = glob(["**"]),
)
