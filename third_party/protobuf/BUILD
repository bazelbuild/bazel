licenses(["notice"])

exports_files(["3.11.3.patch"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),  # glob everything to satisfy the compile.sh srcs test
    visibility = ["//third_party:__pkg__"],
)

alias(
    name = "protoc",
    actual = "@com_google_protobuf//:protoc",
    visibility = ["//visibility:public"],
)

alias(
    name = "protobuf_java",
    actual = "@com_google_protobuf//:protobuf_java",
    visibility = ["//visibility:public"],
)

alias(
    name = "protobuf_java_util",
    actual = "@com_google_protobuf//:protobuf_java_util",
    visibility = ["//visibility:public"],
)

alias(
    name = "protobuf_python",
    actual = "@com_google_protobuf//:protobuf_python",
    visibility = ["//visibility:public"],
)

alias(
    name = "protobuf",
    actual = "@com_google_protobuf//:protobuf",
    visibility = ["//visibility:public"],
)

alias(
    name = "protobuf_headers",
    actual = "@com_google_protobuf//:protobuf_headers",
    visibility = ["//visibility:public"],
)

alias(
    name = "protoc_lib",
    actual = "@com_google_protobuf//:protoc_lib",
    visibility = ["//visibility:public"],
)

alias(
    name = "java_toolchain",
    actual = "@com_google_protobuf//:java_toolchain",
    visibility = ["//visibility:public"],
)
