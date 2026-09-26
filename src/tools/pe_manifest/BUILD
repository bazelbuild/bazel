load("@bazel_pip_dev_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_test")

package(default_applicable_licenses = ["//:license"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

py_binary(
    name = "pe_manifest",
    srcs = ["pe_manifest.py"],
    main = "pe_manifest.py",
    visibility = [
        "//src:__pkg__",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:__pkg__",
    ],
    deps = [
        requirement("grope"),
        requirement("pe-tools"),
    ],
)

py_test(
    name = "pe_manifest_test",
    srcs = [
        "pe_manifest.py",
        "pe_manifest_test.py",
    ],
    main = "pe_manifest_test.py",
    deps = [
        requirement("grope"),
        requirement("pe-tools"),
    ],
)
