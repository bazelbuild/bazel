load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache License 2.0

license(
    name = "license",
    package_name = "kotlin_stdlib",
    license_kinds = [
        "@rules_license//licenses/spdx:Apache-2.0",
    ],
    license_text = "LICENSE",
    package_version = "1.3.21",
)

exports_files(["LICENSE"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "kotlin_stdlib",
    jars = glob(["*.jar"]),
)
