py_binary(
    name = "build_incremental_dexmanifest",
    srcs = [":build_incremental_dexmanifest.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

py_binary(
    name = "build_split_manifest",
    srcs = ["build_split_manifest.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/gflags",
    ],
)

py_test(
    name = "build_split_manifest_test",
    srcs = ["build_split_manifest_test.py"],
    deps = [
        ":build_split_manifest",
    ],
)

py_binary(
    name = "incremental_install",
    srcs = ["incremental_install.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/concurrent:futures",
        "//third_party/py/gflags",
    ],
)

py_test(
    name = "incremental_install_test",
    srcs = ["incremental_install_test.py"],
    deps = [
        ":incremental_install",
        "//third_party/py/mock",
    ],
)

py_binary(
    name = "strip_resources",
    srcs = ["strip_resources.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/gflags",
    ],
)

py_binary(
    name = "stubify_manifest",
    srcs = ["stubify_manifest.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/gflags",
    ],
)

py_test(
    name = "stubify_manifest_test",
    srcs = ["stubify_manifest_test.py"],
    deps = [
        ":stubify_manifest",
    ],
)
