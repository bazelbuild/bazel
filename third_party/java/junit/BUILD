load("@rules_license//rules:license.bzl", "license")

licenses(["notice"])

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "junit",
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
    name = "junit",
    actual = "//third_party:junit4",
)

alias(
    name = "not-testonly",
    actual = "//third_party:junit4",
)

alias(
    name = "not-testonly-android",
    actual = "//third_party:junit4",
)
