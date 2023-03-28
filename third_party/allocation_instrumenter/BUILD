load("@rules_license//rules:license.bzl", "license")

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

java_library(
    name = "allocation_instrumenter",
    exports = ["@maven//:com_google_code_java_allocation_instrumenter_java_allocation_instrumenter"]
)
