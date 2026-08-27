load("@rules_license//rules:license.bzl", "license")

licenses(["notice"])

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "grpc",
    license_kinds = [
        "@rules_license//licenses/spdx:Apache-2.0",
    ],
    license_text = "LICENSE",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

alias(
    name = "auth",
    actual = "//third_party/grpc-java:grpc-jar",
)

alias(
    name = "context",
    actual = "//third_party/grpc-java:grpc-jar",
)

alias(
    name = "core",
    actual = "//third_party/grpc-java:grpc-jar",
)

alias(
    name = "inprocess",
    actual = "//third_party/grpc-java:grpc-jar",
)

alias(
    name = "netty",
    actual = "//third_party/grpc-java:grpc-jar",
)

alias(
    name = "protobuf",
    actual = "//third_party/grpc-java:grpc-jar",
)

alias(
    name = "stub",
    actual = "//third_party/grpc-java:grpc-jar",
)

alias(
    name = "util",
    actual = "//third_party/grpc-java:grpc-jar",
)
