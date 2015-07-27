licenses(["notice"])  # Apache 2.0

TEST_DATA = [
    "//tools/build_defs/docker/testdata:base_with_entrypoint",
    "//tools/build_defs/docker/testdata:base_with_volume",
    "//tools/build_defs/docker/testdata:derivative_with_cmd",
    "//tools/build_defs/docker/testdata:derivative_with_shadowed_cmd",
    "//tools/build_defs/docker/testdata:derivative_with_volume",
    "//tools/build_defs/docker/testdata:files_base",
    "//tools/build_defs/docker/testdata:files_with_files_base",
    "//tools/build_defs/docker/testdata:files_with_tar_base",
    "//tools/build_defs/docker/testdata:tar_base",
    "//tools/build_defs/docker/testdata:tar_with_files_base",
    "//tools/build_defs/docker/testdata:tar_with_tar_base",
    "//tools/build_defs/docker/testdata:generated_tarball",
    "//tools/build_defs/docker/testdata:with_env",
    "//tools/build_defs/docker/testdata:with_double_env",
    "//tools/build_defs/docker/testdata:link_with_files_base",
]

sh_test(
    name = "build_test",
    size = "medium",
    srcs = [
        "build_test.sh",
    ],
    data = TEST_DATA + ["testenv.sh"],
    deps = [
        "//src/test/shell:bashunit",
    ],
)

# Used by docker_build and friends
py_library(
    name = "archive",
    srcs = ["archive.py"],
    visibility = ["//tools/build_defs/docker:__subpackages__"],
)

py_binary(
    name = "rewrite_json",
    srcs = ["rewrite_json.py"],
    visibility = ["//visibility:public"],
    deps = ["//third_party/py/gflags"],
)

py_test(
    name = "rewrite_json_test",
    srcs = ["rewrite_json_test.py"],
    deps = [
        ":rewrite_json",
    ],
)

py_binary(
    name = "build_layer",
    srcs = ["build_layer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":archive",
        "//third_party/py/gflags",
    ],
)

py_binary(
    name = "sha256",
    srcs = ["sha256.py"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "create_image",
    srcs = ["create_image.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":archive",
        "//third_party/py/gflags",
    ],
)
