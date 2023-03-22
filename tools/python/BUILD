load("@rules_python//python:defs.bzl", "py_test")
load(":utils.bzl", "expand_pyversion_template")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/python/runfiles:srcs",
    ],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["**"]) + [
        "//tools/python/runfiles:embedded_tools",
    ],
    visibility = ["//tools:__pkg__"],
)

expand_pyversion_template(
    name = "_generate_wrappers",
    out2 = ":py2wrapper.sh",
    out2_nonstrict = ":py2wrapper_nonstrict.sh",
    out3 = ":py3wrapper.sh",
    out3_nonstrict = ":py3wrapper_nonstrict.sh",
    template = "pywrapper_template.txt",
    visibility = ["//visibility:private"],
)

py_test(
    name = "pywrapper_test",
    srcs = ["pywrapper_test.py"],
    data = [
        ":py3wrapper.sh",
        ":py3wrapper_nonstrict.sh",
    ],
    deps = ["//src/test/py/bazel:test_base"],
)
