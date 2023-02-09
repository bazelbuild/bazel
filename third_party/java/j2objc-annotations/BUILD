licenses(["notice"])  # Apache License 2.0

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

java_library(
    name = "j2objc-annotations",
    exports = ["@maven//:com_google_j2objc_j2objc_annotations"],
)
