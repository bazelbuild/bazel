
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

java_import(
    name = "j2objc-annotations",
    jars = [":j2objc-annotations-1.1.jar"],
)
