load("@rules_license//rules:license.bzl", "license")

licenses(["notice"])

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "fastutil",
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
    name = "int-array-list",
    actual = "//third_party:fastutil",
)

alias(
    name = "int-arrays",
    actual = "//third_party:fastutil",
)

alias(
    name = "int-to-object-open-hash-map",
    actual = "//third_party:fastutil",
)

alias(
    name = "object-to-int-open-hash-map",
    actual = "//third_party:fastutil",
)

alias(
    name = "reference-to-int-open-hash-map",
    actual = "//third_party:fastutil",
)
