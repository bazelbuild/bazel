load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache License 2.0

license(
    name = "license",
    package_name = "jetbrains_annotations/13.0",
    license_kinds = [
        "@rules_license//licenses/spdx:Apache-2.0",
    ],
    license_text = "LICENSE",
)

exports_files(["LICENSE"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "jetbrains_annotations",
    jars = glob(["*.jar"]),
)
