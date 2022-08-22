load("@rules_java//java:defs.bzl", "java_proto_library")
load("@rules_license//rules:license.bzl", "license")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

license(
    name = "license",
    package_name = "pprof",
    license_kinds = [
        "@rules_license//licenses/spdx:Apache-2.0",
    ],
    license_text = "LICENSE",
)

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
