load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

license(
    name = "license",
    copyright_notice = "Copyright (c) 2008 Kohsuke Kawaguchi and codehaus.org",
    license_kinds = [
        "@rules_license//licenses/spdx:MIT",
    ],
    license_text = "LICENSE",
    package_name = "annimal_sniffer",
    package_version = "1.14",
)

exports_files(["LICENSE"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "animal_sniffer",
    jars = ["animal-sniffer-annotations-1.14.jar"],
)
