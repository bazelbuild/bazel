licenses(["notice"])  # Apache License 2.0

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

alias(
    name = "javapoet",
    actual = "@maven//:com_squareup_javapoet",
)
