load("@rules_java//java:defs.bzl", "java_library")
load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

license(
    name = "license",
    package_name = "jformatstring",
    license_kinds = [
        "@rules_license//licenses/spdx:GPL-2.0-with-classpath-exception",
    ],
    license_text = "LICENSE",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "jformatstring",
    srcs = glob(["java/**"]),
)

load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library")

bootstrap_java_library(
    name = "bootstrap",
    srcs = glob(["java/**"]),
)
