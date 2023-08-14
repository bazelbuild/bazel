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

alias(
    name = "j2objc-annotations",
    actual = "@maven//:com_google_j2objc_j2objc_annotations",
)
