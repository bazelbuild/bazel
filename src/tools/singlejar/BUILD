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
    name = "zlib_interface",
    srcs = ["zlib_interface.h"],
)
