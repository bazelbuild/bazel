load("@rules_license//rules:license.bzl", "license")

package(
    default_visibility = [
        "//src/tools/android/java/com/google/devtools/build/android/dexer:__pkg__",
    ],
    default_applicable_licenses = [":license"],
)

licenses(["notice"])

filegroup(
  name = "srcs",
  srcs = glob(["**"]),
  visibility = ["//third_party:__pkg__"],
)

java_library(
    name = "android_dex",
    srcs = glob(["java/**/*.java"]),
)

license(
    name = "license",
    package_name = "android_dex",
    license_kinds = [
        "@rules_license//licenses/spdx:Apache-2.0",
    ],
    license_text = "NOTICE",
    package_version = "d8@cl.451207830",
)
