licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//tools/build_defs/docker/testdata:srcs"],
    visibility = ["//tools:__pkg__"],
)

exports_files(
    ["README.md"],
    visibility = ["//site:__pkg__"],
)

TEST_TARGETS = [
    "base_with_entrypoint",
    "base_with_volume",
    "derivative_with_cmd",
    "derivative_with_shadowed_cmd",
    "derivative_with_volume",
    "directory_with_tar_base",
    "files_base",
    "files_with_files_base",
    "files_with_tar_base",
    "tar_base",
    "tar_with_files_base",
    "tar_with_tar_base",
    "generated_tarball",
    "with_env",
    "with_double_env",
    "with_label",
    "with_double_label",
    "with_user",
    "workdir_with_tar_base",
    "link_with_files_base",
]

TEST_DATA = [
    "//tools/build_defs/docker/testdata:%s.tar" % t
    for t in TEST_TARGETS
] + [
    "//tools/build_defs/docker/testdata:notop_%s.tar" % t
    for t in TEST_TARGETS
] + [
    "//tools/build_defs/docker/testdata:gen_image.tar",
    "//tools/build_defs/docker/testdata:data_path_image.tar",
    "//tools/build_defs/docker/testdata:no_data_path_image.tar",
    "//tools/build_defs/docker/testdata:absolute_data_path_image.tar",
    "//tools/build_defs/docker/testdata:root_data_path_image.tar",
    "//tools/build_defs/docker/testdata:dummy_repository.tar",
    "//tools/build_defs/docker/testdata:extras_with_deb.tar",
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

sh_test(
    name = "build_test_oci",
    size = "medium",
    srcs = [
        "build_test_oci.sh",
    ],
    data = TEST_DATA + ["testenv.sh"],
    deps = [
        "//src/test/shell:bashunit",
    ],
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
    name = "sha256",
    srcs = ["sha256.py"],
    visibility = ["//visibility:public"],
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
)

filegroup(
    name = "incremental_load_template",
    srcs = ["incremental_load.sh.tpl"],
    visibility = ["//visibility:public"],
)
