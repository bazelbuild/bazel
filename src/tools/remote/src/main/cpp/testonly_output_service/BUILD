load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

cc_binary(
    name = "testonly_output_service",
    srcs = [
        "bazel_output_service_impl.cc",
        "bazel_output_service_impl.h",
        "main.cc",
        "memory.cc",
        "memory.h",
        "memory_unix.cc",
        "string.cc",
        "string.h",
    ],
    deps = [
        "//src/main/protobuf:bazel_output_service_cc_grpc",
        "//src/main/protobuf:bazel_output_service_cc_proto",
        "//src/main/protobuf:bazel_output_service_rev2_cc_proto",
        "//third_party/grpc:grpc++_unsecure",
    ],
)
