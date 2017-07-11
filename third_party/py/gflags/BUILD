licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "gflags",
    deps = ["//third_party/py/six"],
    srcs = glob(["**/*.py"]),
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)
