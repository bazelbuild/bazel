licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//tools/build_defs/docker/testdata:srcs"],
    visibility = ["//tools:__pkg__"],
)

TEST_TARGETS = [
    "base_with_entrypoint",
    "base_with_volume",
    "derivative_with_cmd",
    "derivative_with_shadowed_cmd",
    "derivative_with_volume",
    "files_base",
    "files_with_files_base",
    "files_with_tar_base",
    "tar_base",
    "tar_with_files_base",
    "tar_with_tar_base",
    "generated_tarball",
    "with_env",
    "with_double_env",
    "workdir_with_tar_base",
    "link_with_files_base",
]

TEST_DATA = [
    "//tools/build_defs/docker/testdata:" + t
    for t in TEST_TARGETS
] + [
    "//tools/build_defs/docker/testdata:notop_" + t
    for t in TEST_TARGETS
] + [
    "//tools/build_defs/docker/testdata:gen_image",
    "//tools/build_defs/docker/testdata:data_path_image",
    "//tools/build_defs/docker/testdata:no_data_path_image",
    "//tools/build_defs/docker/testdata:dummy_repository",
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

py_test(
    name = "archive_test",
    srcs = [
        "archive_test.py",
        "testenv.py",
    ],
    data = ["//tools/build_defs/docker/testdata:archive_testdata"],
    deps = [":archive"],
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
