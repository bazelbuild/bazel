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

test_suite(
    name = "all_windows_tests",
    tests = [
        "//tools/python/runfiles:all_windows_tests",
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
        ":py2wrapper.sh",
        ":py2wrapper_nonstrict.sh",
    ],
    python_version = "PY2",
    deps = ["//src/test/py/bazel:test_base"],
)
