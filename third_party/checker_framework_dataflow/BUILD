load("@rules_license//rules:license.bzl", "license")
load("@rules_java//java:defs.bzl", "java_import")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

license(
    name = "license",
    copyright_notice = "Copyright © 2014 The Bazel Authors. All rights reserved.",
    license_kinds = [
        "@rules_license//licenses/spdx:GPL-2.0-with-classpath-exception",
        "@rules_license//licenses/spdx:MIT",
    ],
    license_text = "LICENSE.txt",
    package_name = "dataflow-errorprone",
    package_version = "3.15.0",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "checker_framework_dataflow",
    jars = ["dataflow-errorprone-3.15.0.jar"],
    srcjar = "dataflow-errorprone-3.15.0-sources.jar",
)

filegroup(
    name = "bootstrap",
    srcs = ["dataflow-errorprone-3.15.0.jar"],
)
