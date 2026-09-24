load("@rules_license//rules:license.bzl", "license")

licenses(["notice"])

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "netty",
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
    name = "buffer",
    actual = "//third_party:netty",
)

alias(
    name = "codec",
    actual = "//third_party:netty",
)

alias(
    name = "codec_http",
    actual = "//third_party:netty",
)

alias(
    name = "common",
    actual = "//third_party:netty",
)

alias(
    name = "handler",
    actual = "//third_party:netty",
)

alias(
    name = "transport",
    actual = "//third_party:netty",
)

alias(
    name = "transport_native_epoll",
    actual = "//third_party:netty",
)

alias(
    name = "transport_native_kqueue",
    actual = "//third_party:netty",
)

alias(
    name = "transport_native_unix_common",
    actual = "//third_party:netty",
)
