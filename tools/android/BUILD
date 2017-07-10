package(default_visibility = ["//tools:__pkg__"])

py_binary(
    name = "merge_manifests",
    srcs = [
        "android_permissions.py",
        "merge_manifests.py",
    ],
    deps = [
        "//third_party/py/gflags",
    ],
)

py_test(
    name = "merge_manifests_test",
    srcs = ["merge_manifests_test.py"],
    deps = [":merge_manifests"],
)

py_binary(
    name = "build_incremental_dexmanifest",
    srcs = [":build_incremental_dexmanifest.py"],
)

sh_test(
    name = "build_incremental_dexmanifest_test",
    srcs = [
        "build_incremental_dexmanifest_test.sh",
    ],
    args = ["$(location //tools/zip:zipper)"],
    data = [
        ":build_incremental_dexmanifest",
        "//tools/zip:zipper",
    ],
)

py_binary(
    name = "build_split_manifest",
    srcs = ["build_split_manifest.py"],
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
    deps = ["//third_party/py/gflags"],
)

sh_test(
    name = "strip_resources_test",
    srcs = ["strip_resources_test.sh"],
    args = ["$(location //tools/zip:zipper)"],
    data = [
        ":strip_resources",
        "//tools/zip:zipper",
    ],
)

py_binary(
    name = "aar_native_libs_zip_creator",
    srcs = [
        "aar_native_libs_zip_creator.py",
    ],
    deps = [
        "//third_party/py/gflags",
    ],
)

py_test(
    name = "aar_native_libs_zip_creator_test",
    srcs = [
        "aar_native_libs_zip_creator_test.py",
    ],
    deps = [
        ":aar_native_libs_zip_creator",
    ],
)

py_binary(
    name = "stubify_manifest",
    srcs = ["stubify_manifest.py"],
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

sh_binary(
    name = "zip_manifest_creator",
    srcs = ["zip_manifest_creator.sh"],
    data = ["//tools/zip:zipper"],
)

sh_test(
    name = "zip_manifest_creator_test",
    size = "small",
    srcs = ["zip_manifest_creator_test.sh"],
    args = ["$(location //tools/zip:zipper)"],
    data = [
        ":zip_manifest_creator",
        "//tools/zip:zipper",
    ],
)

py_binary(
    name = "aar_embedded_jars_extractor",
    srcs = ["aar_embedded_jars_extractor.py"],
    deps = ["//third_party/py/gflags"],
)

py_test(
    name = "aar_embedded_jars_extractor_test",
    srcs = ["aar_embedded_jars_extractor_test.py"],
    deps = [":aar_embedded_jars_extractor"],
)

py_binary(
    name = "resource_extractor",
    srcs = ["resource_extractor.py"],
)

py_test(
    name = "resource_extractor_test",
    srcs = ["resource_extractor_test.py"],
    deps = [":resource_extractor"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//tools/android/emulator:srcs"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(
        ["*"],
        exclude = [
            "*_test.py",
            "*_test.sh",
        ],
    ) + ["//tools/android/emulator:embedded_tools"],
)

# Exported for AndroidSdkRepositoryTest to use it instead of mocking it out.
exports_files(["android_sdk_repository_template.bzl"])
