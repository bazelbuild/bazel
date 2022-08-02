load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

license(
    name = "license",
    package_name = "antlr4",
    copyright_notice = "Copyright (c) 2012 Terence Parr and Sam Harwell",
    license_kinds = [
        "@rules_license//licenses/spdx:0BSD",
    ],
    license_text = "LICENSE",
    package_version = "4.5.3",
)

exports_files(["LICENSE"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "antlr",
    jars = ["antlr4-4.5.3.jar"],
)
