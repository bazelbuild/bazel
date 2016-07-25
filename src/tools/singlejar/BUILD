# Description:
#   singlejar C++ implementation.
package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

cc_test(
    name = "input_jar_scan_jartool_test",
    size = "large",
    srcs = [
        "input_jar_scan_entries_test.h",
        "input_jar_scan_jartool_test.cc",
        ":input_jar",
        ":test_util",
    ],
    copts = ["-DJAR_TOOL_PATH=\\\"external/local_jdk/bin/jar\\\""],
    data = [
        "//external:jar",
        "//external:jdk-default",
    ],
    # Timing out, see https://github.com/bazelbuild/bazel/issues/1555
    tags = ["manual"],
    deps = ["//third_party:gtest"],
)

cc_test(
    name = "input_jar_scan_ziptool_test",
    size = "large",
    srcs = [
        "input_jar_scan_entries_test.h",
        "input_jar_scan_ziptool_test.cc",
        ":input_jar",
        ":test_util",
    ],
    # Timing out, see https://github.com/bazelbuild/bazel/issues/1555
    tags = ["manual"],
    deps = ["//third_party:gtest"],
)

cc_test(
    name = "input_jar_bad_jar_test",
    srcs = [
        "input_jar_bad_jar_test.cc",
        ":input_jar",
        ":test_util",
    ],
    deps = ["//third_party:gtest"],
)

cc_test(
    name = "options_test",
    srcs = [
        "options.h",
        "options_test.cc",
        ":token_stream",
    ],
    deps = [
        ":options",
        "//third_party:gtest",
    ],
)

cc_test(
    name = "token_stream_test",
    srcs = [
        "token_stream_test.cc",
        ":token_stream",
    ],
    deps = ["//third_party:gtest"],
)

cc_test(
    name = "transient_bytes_test",
    size = "large",
    srcs = [
        "transient_bytes_test.cc",
        ":input_jar",
        ":test_util",
        ":transient_bytes",
        ":zlib_interface",
    ],
    # Timing out, see https://github.com/bazelbuild/bazel/issues/1555
    tags = ["manual"],
    deps = [
        "//third_party:gtest",
        "//third_party/zlib",
    ],
)

cc_test(
    name = "zip_headers_test",
    size = "small",
    srcs = [
        "zip_headers_test.cc",
        ":zip_headers",
    ],
    deps = ["//third_party:gtest"],
)

cc_test(
    name = "zlib_interface_test",
    srcs = [
        "zlib_interface_test.cc",
        ":zlib_interface",
    ],
    deps = [
        "//third_party:gtest",
        "//third_party/zlib",
    ],
)

cc_library(
    name = "options",
    srcs = [
        "diag.h",
        "options.cc",
        "options.h",
        ":token_stream",
    ],
)

filegroup(
    name = "input_jar",
    srcs = [
        "diag.h",
        "input_jar.h",
        "mapped_file.h",
        "zip_headers.h",
    ],
)

filegroup(
    name = "test_util",
    srcs = ["test_util.h"],
)

filegroup(
    name = "token_stream",
    srcs = [
        "diag.h",
        "token_stream.h",
    ],
)

filegroup(
    name = "transient_bytes",
    srcs = [
        "diag.h",
        "transient_bytes.h",
        "zlib_interface.h",
        ":zip_headers",
    ],
)

filegroup(
    name = "zip_headers",
    srcs = ["zip_headers.h"],
)

filegroup(
    name = "zlib_interface",
    srcs = [
        "diag.h",
        "zlib_interface.h",
    ],
)
