licenses(["notice"])  #  Apache-2 license

filegroup(
    name = "srcs",
    srcs = glob(["**/*"]),
    visibility = ["//third_party:__pkg__"],
)

java_library(
    name = "aws-sdk-auth-lite",
    srcs = glob(["**/*.java"]),
    visibility = ["//visibility:public"],
    deps = [
        "//third_party:apache_commons_logging",
        "//third_party:gson",
        "//third_party:guava",
    ],
)
