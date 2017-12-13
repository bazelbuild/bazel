# Protos of the Actions for Android rules.

package(default_visibility = ["//visibility:public"])

java_proto_library(
    name = "serialize_format_java_pb",
    deps = [":serialize_format_proto"],
)

proto_library(
    name = "serialize_format_proto",
    srcs = ["serialize_format.proto"],
)

java_proto_library(
    name = "resources_java_proto",
    strict_deps = 0,
    deps = [":resources_proto"],
)

proto_library(
    name = "resources_proto",
    srcs = [
        "Configuration.proto",
        "Resources.proto",
        "ResourcesInternal.proto",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
