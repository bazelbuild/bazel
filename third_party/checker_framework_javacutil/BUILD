load("@rules_java//java:defs.bzl", "java_import")
load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

license(
    name = "license",
    license_kinds = [
        "@rules_license//licenses/spdx:GPL-2.0-with-classpath-exception",
    ],
    license_text = "LICENSE.txt",
    package_name = "javacutil",
    package_version = "3.2.0",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "checker_framework_javacutil",
    jars = ["javacutil-3.2.0.jar"],
    srcjar = "javacutil-3.2.0-sources.jar",
)

filegroup(
    name = "bootstrap",
    srcs = ["javacutil-3.2.0.jar"],
)
