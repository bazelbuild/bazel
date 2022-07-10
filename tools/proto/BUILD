package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"] + ["**/*"]),
)

exports_files(["BUILD"])

alias(
    name = "protoc",
    actual = "@com_google_protobuf//:protoc",
)

alias(
    name = "javalite_toolchain",
    actual = "@com_google_protobuf//:javalite_toolchain",
)

alias(
    name = "java_toolchain",
    actual = "@com_google_protobuf//:java_toolchain",
)

alias(
    name = "cc_toolchain",
    actual = "@com_google_protobuf//:cc_toolchain",
)
