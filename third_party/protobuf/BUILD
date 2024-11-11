licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),  # glob everything to satisfy the compile.sh srcs test
    visibility = ["//third_party:__pkg__"],
)

alias(
    name = "protoc",
    actual = "@protobuf//:protoc",
    visibility = ["//visibility:public"],
)

alias(
    name = "protobuf_java",
    actual = "@protobuf//:protobuf_java",
    visibility = ["//visibility:public"],
)

alias(
    name = "protobuf_java_util",
    actual = "@protobuf//:protobuf_java_util",
    visibility = ["//visibility:public"],
)

alias(
    name = "protobuf_python",
    actual = "@protobuf//:protobuf_python",
    visibility = ["//visibility:public"],
)

alias(
    name = "protoc_lib",
    actual = "@protobuf//:protoc_lib",
    visibility = ["//visibility:public"],
)
