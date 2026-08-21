load("@rules_license//rules:license.bzl", "license")

licenses(["notice"])

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "guava",
    license_kinds = [
        "@rules_license//licenses/spdx:Apache-2.0",
    ],
    license_text = "LICENSE",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

alias(
    name = "annotations",
    actual = "//third_party:guava",
)

alias(
    name = "base",
    actual = "//third_party:guava",
)

alias(
    name = "cache",
    actual = "//third_party:guava",
)

alias(
    name = "collect",
    actual = "//third_party:guava",
)

alias(
    name = "escape",
    actual = "//third_party:guava",
)

alias(
    name = "eventbus",
    actual = "//third_party:guava",
)

alias(
    name = "graph",
    actual = "//third_party:guava",
)

alias(
    name = "hash",
    actual = "//third_party:guava",
)

alias(
    name = "html",
    actual = "//third_party:guava",
)

alias(
    name = "htmlescapers",
    actual = "//third_party:guava",
)

alias(
    name = "io",
    actual = "//third_party:guava",
)

alias(
    name = "math",
    actual = "//third_party:guava",
)

alias(
    name = "net",
    actual = "//third_party:guava",
)

alias(
    name = "primitives",
    actual = "//third_party:guava",
)

alias(
    name = "reflect",
    actual = "//third_party:guava",
)

alias(
    name = "testing",
    actual = "//third_party:guava-testlib",
)

alias(
    name = "util",
    actual = "//third_party:guava",
)

alias(
    name = "concurrent",
    actual = "//third_party:guava",
)

alias(
    name = "collect-testing",
    actual = "//third_party:guava-testlib",
)

alias(
    name = "testlib",
    actual = "//third_party:guava-testlib",
)
