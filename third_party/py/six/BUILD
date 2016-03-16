licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

py_library(
    name = "six",
    srcs = ["six.py"],
    imports = ["."],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)
