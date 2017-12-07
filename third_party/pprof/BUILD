package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

proto_library(
    name = "profile_proto",
    srcs = ["profile.proto"],
)

java_proto_library(
    name = "profile_java_proto",
    deps = [":profile_proto"],
)
