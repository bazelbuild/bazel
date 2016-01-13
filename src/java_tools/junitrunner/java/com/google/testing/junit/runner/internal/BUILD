package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "internal",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:guava",
        "//third_party:guice",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:junit4",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
