licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "abseil",
    deps = ["//third_party/py/six"],
    srcs = glob(["**/*.py"]),
    srcs_version = "PY2AND3",
    imports = ["."],
    visibility = ["//visibility:public"],
)
