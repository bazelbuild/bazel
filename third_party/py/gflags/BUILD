licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "gflags",
    srcs = [
        "__init__.py",
        "gflags_validators.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)
