package(default_visibility = ["//visibility:public"])

filegroup(name = "sdk")

filegroup(
    name = "incremental_stub_application",
    srcs = [],
)

filegroup(
    name = "incremental_split_stub_application",
    srcs = [],
)

filegroup(
    name = "aar_generator",
    srcs = ["//src/tools/android/java/com/google/devtools/build/android:AarGeneratorAction"],
)

filegroup(
    name = "resources_processor",
    srcs = ["//src/tools/android/java/com/google/devtools/build/android:AndroidResourceProcessingAction"],
)

sh_binary(
    name = "merge_dexzips",
    srcs = ["fail.sh"],
)

sh_binary(
    name = "shuffle_jars",
    srcs = ["fail.sh"],
)

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
    name = "proguard_whitelister",
    srcs = [
        "proguard_whitelister.py",
    ],
    deps = [
        "//third_party/py/gflags",
    ],
)

py_test(
    name = "proguard_whitelister_test",
    srcs = ["proguard_whitelister_test.py"],
    data = ["proguard_whitelister_input.cfg"],
    deps = [
        ":proguard_whitelister",
    ],
)

py_binary(
    name = "build_incremental_dexmanifest",
    srcs = [":build_incremental_dexmanifest.py"],
    deps = [],
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
    deps = [
        "//third_party/py/gflags",
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

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
