package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

filegroup(
    name = "test-deps",
    testonly = 1,
    srcs = ["//src:bazel"],
    data = ["//src/tools/remote:worker"],
)

py_library(
    name = "test_base",
    testonly = 1,
    srcs = ["test_base.py"],
    data = [":test-deps"],
    visibility = [
        "//src/test/py/bazel:__pkg__",
        "//third_party/def_parser:__pkg__",
        "//tools/android:__pkg__",
        "//tools/build_rules:__pkg__",
    ],
)

py_test(
    name = "action_temp_test",
    size = "medium",
    srcs = ["action_temp_test.py"],
    deps = [":test_base"],
)

py_test(
    name = "cc_import_test",
    size = "medium",
    srcs = ["cc_import_test.py"],
    deps = [":test_base"],
)

py_test(
    name = "py_test",
    size = "medium",
    srcs = ["py_test.py"],
    deps = [":test_base"],
)

py_test(
    name = "bazel_server_mode_test",
    size = "medium",
    srcs = ["bazel_server_mode_test.py"],
    deps = [":test_base"],
)

py_test(
    name = "bazel_clean_test",
    size = "medium",
    srcs = ["bazel_clean_test.py"],
    deps = [":test_base"],
)

py_test(
    name = "bazel_external_repository_test",
    size = "medium",
    srcs = ["bazel_external_repository_test.py"],
    data = glob(["testdata/bazel_external_repository_test/**"]),
    deps = [":test_base"],
)

py_test(
    name = "bazel_windows_test",
    size = "medium",
    srcs = select({
        "//src/conditions:windows": ["bazel_windows_test.py"],
        "//conditions:default": ["empty_test.py"],
    }),
    main = select({
        "//src/conditions:windows": "bazel_windows_test.py",
        "//conditions:default": "empty_test.py",
    }),
    deps = select({
        "//src/conditions:windows": [":test_base"],
        "//conditions:default": [],
    }),
)

py_test(
    name = "windows_remote_test",
    size = "medium",
    srcs = select({
        "//src/conditions:windows": ["windows_remote_test.py"],
        "//conditions:default": ["empty_test.py"],
    }),
    main = select({
        "//src/conditions:windows": "windows_remote_test.py",
        "//conditions:default": "empty_test.py",
    }),
    deps = select({
        "//src/conditions:windows": [":test_base"],
        "//conditions:default": [],
    }),
)

py_test(
    name = "launcher_test",
    size = "medium",
    srcs = ["launcher_test.py"],
    deps = [":test_base"],
)

py_test(
    name = "runfiles_test",
    timeout = "long",
    srcs = ["runfiles_test.py"],
    data = glob(["testdata/runfiles_test/**"]),
    deps = [
        ":test_base",
        "//third_party/py/six",
    ],
)

py_test(
    name = "runfiles_sandboxed_test",
    timeout = "long",
    srcs = ["runfiles_sandboxed_test.py"],
    data = glob(["testdata/runfiles_test/**"]),
    tags = [
        # Windows does not support sandboxing yet.
        "no_windows",
    ],
    deps = [":test_base"],
)

py_test(
    name = "bazel_windows_cpp_test",
    size = "large",
    srcs = select({
        "//src/conditions:windows": ["bazel_windows_cpp_test.py"],
        "//conditions:default": ["empty_test.py"],
    }),
    main = select({
        "//src/conditions:windows": "bazel_windows_cpp_test.py",
        "//conditions:default": "empty_test.py",
    }),
    deps = select({
        "//src/conditions:windows": [":test_base"],
        "//conditions:default": [],
    }),
)

py_test(
    name = "test_wrapper_test",
    srcs = select({
        "//src/conditions:windows": ["test_wrapper_test.py"],
        "//conditions:default": ["empty_test.py"],
    }),
    main = select({
        "//src/conditions:windows": "test_wrapper_test.py",
        "//conditions:default": "empty_test.py",
    }),
    deps = select({
        "//src/conditions:windows": [":test_base"],
        "//conditions:default": [],
    }),
)

py_test(
    name = "query_test",
    size = "medium",
    srcs = ["query_test.py"],
    deps = [":test_base"],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
    visibility = ["//src:__pkg__"],
)
