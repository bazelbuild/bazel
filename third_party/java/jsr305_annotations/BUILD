load("@rules_license//rules:license.bzl", "license")

licenses(["notice"])

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "jsr305",
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
    name = "jsr305_annotations",
    actual = "//third_party:jsr305",
)

alias(
    name = "jsr305.jar",
    actual = "//third_party:jsr305-jars",
)
