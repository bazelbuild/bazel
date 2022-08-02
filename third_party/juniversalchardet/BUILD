load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

licenses(["reciprocal"])  # MPL 1.1

license(
    name = "license",
    package_name = "juniversalchardet",
    license_kinds = [
        "@rules_license//licenses/spdx:MPL-1.1",
    ],
    license_text = "LICENSE",
    package_version = "1.0.3",
)

exports_files(["LICENSE"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "juniversalchardet",
    jars = glob(["*.jar"]),
)
