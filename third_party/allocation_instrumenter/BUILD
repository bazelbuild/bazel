load("@rules_license//rules:license.bzl", "license")
load("//tools/distributions:distribution_rules.bzl", "distrib_java_import")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

license(
    name = "license",
    package_name = "allocation_instrumenter",
    license_kinds = [
        "@rules_license//licenses/spdx:Apache-2.0",
    ],
    license_text = "LICENSE",
    package_version = "3.3.0",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

distrib_java_import(
    name = "allocation_instrumenter",
    enable_distributions = ["debian"],
    jars = ["java-allocation-instrumenter-3.3.0.jar"],
)
