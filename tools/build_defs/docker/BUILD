licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools:__pkg__"],
)

exports_files(
    ["README.md"],
    visibility = ["//site:__pkg__"],
)

# Used by docker_build and friends

py_binary(
    name = "rewrite_json",
    srcs = ["rewrite_json.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//third_party/py/gflags",
    ],
)

py_test(
    name = "rewrite_json_test",
    srcs = ["rewrite_json_test.py"],
    deps = [
        ":rewrite_json",
    ],
)

py_binary(
    name = "create_image_config",
    srcs = ["create_image_config.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//third_party/py/gflags",
    ],
)

py_test(
    name = "create_image_config_test",
    srcs = ["create_image_config_test.py"],
    deps = [
        ":create_image_config",
    ],
)

py_binary(
    name = "create_image",
    srcs = ["create_image.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//third_party/py/gflags",
        "//tools/build_defs/pkg:archive",
    ],
)

py_binary(
    name = "join_layers",
    srcs = ["join_layers.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//third_party/py/gflags",
        "//tools/build_defs/pkg:archive",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//third_party/py/six",
    ],
)

filegroup(
    name = "incremental_load_template",
    srcs = ["incremental_load.sh.tpl"],
    visibility = ["//visibility:public"],
)
