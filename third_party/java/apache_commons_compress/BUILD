load("@rules_license//rules:license.bzl", "license")

licenses(["notice"])

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "apache_commons_compress",
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
    name = "apache_commons_compress",
    actual = "//third_party:apache_commons_compress",
)

alias(
    name = "ar",
    actual = "//third_party:apache_commons_compress",
)

alias(
    name = "brotli",
    actual = "//third_party:apache_commons_compress",
)

alias(
    name = "bzip2",
    actual = "//third_party:apache_commons_compress",
)

alias(
    name = "core",
    actual = "//third_party:apache_commons_compress",
)

alias(
    name = "gzip",
    actual = "//third_party:apache_commons_compress",
)

alias(
    name = "sevenz",
    actual = "//third_party:apache_commons_compress",
)

alias(
    name = "tar",
    actual = "//third_party:apache_commons_compress",
)

alias(
    name = "xz",
    actual = "//third_party:apache_commons_compress",
)

alias(
    name = "zstandard",
    actual = "//third_party:apache_commons_compress",
)
