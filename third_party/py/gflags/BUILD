licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

py_library(
    name = "gflags",
    srcs = [
        "__init__.py",
        "gflags_validators.py",
    ],
    visibility = ["//visibility:public"],
)
