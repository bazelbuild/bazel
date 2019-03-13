package(
    default_visibility = [
        "//src:__subpackages__",
        "//third_party/ijar:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

cc_library(
    name = "zip",
    srcs = [
        "zip.cc",
    ] + select({
        "//src:windows": [
            "mapped_file_windows.cc",
        ],
        "//conditions:default": [
            "mapped_file_unix.cc",
        ],
    }),
    hdrs = [
        "common.h",
        "mapped_file.h",
        "zip.h",
    ],
    visibility = [
        "//src:__subpackages__",
        "//third_party/ijar:__subpackages__",
        "//tools/test:__pkg__",
    ],
    deps = [
        ":platform_utils",
        ":zlib_client",
    ] + select({
        "//src:windows": [
            "//src/main/cpp/util:errors",
            "//src/main/cpp/util:filesystem",
            "//src/main/cpp/util:logging",
            "//src/main/cpp/util:strings",
        ],
        "//conditions:default": [
        ],
    }),
)

cc_library(
    name = "zlib_client",
    srcs = ["zlib_client.cc"],
    hdrs = [
        "common.h",
        "zlib_client.h",
    ],
    deps = ["//third_party/zlib"],
)

cc_library(
    name = "platform_utils",
    srcs = ["platform_utils.cc"],
    hdrs = [
        "common.h",
        "platform_utils.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "//src/main/cpp/util:errors",
        "//src/main/cpp/util:filesystem",
        "//src/main/cpp/util:logging",
    ],
)

cc_binary(
    name = "zipper",
    srcs = ["zip_main.cc"],
    visibility = ["//visibility:public"],
    deps = [":zip"],
)

cc_binary(
    name = "ijar",
    srcs = [
        "classfile.cc",
        "ijar.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [":zip"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//third_party/ijar/test:srcs"],
    visibility = ["//third_party:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "ijar_with_deps_archive",
    extension = "tar.gz",
    mode = "755",
    visibility = ["//third_party:__pkg__"],
    deps = [
        ":ijar_archive",
        "//src/main/cpp/util:cpp_util_with_deps_archive",
    ],
)

genrule(
    name = "ijar_with_deps_zip",
    srcs = [
        ":ijar_srcs_zip",
        "//src/main/cpp/util:cpp_util_with_deps_zip",
    ],
    outs = ["ijar_srcs_with_deps.zip"],
    cmd = "$(location //third_party:merge_zip_files.sh) nodir $(OUTS) $(SRCS)",
    tools = ["//third_party:merge_zip_files.sh"],
    visibility = ["//third_party:__pkg__"],
)

IJAR_SOURCES_AND_BINARIES = glob(
    ["**"],
    exclude = ["BUILD"],
) + [
    ":ijar",
    ":zipper",
]

IJAR_ARCHIVE_PACKAGE_DIR = "ijar"

genrule(
    name = "ijar_srcs_zip",
    srcs = IJAR_SOURCES_AND_BINARIES,
    outs = ["ijar_srcs.zip"],
    cmd = "$(location //third_party:zip_java_tools.sh) " + IJAR_ARCHIVE_PACKAGE_DIR + " $(OUTS) $(SRCS)",
    tools = ["//third_party:zip_java_tools.sh"],
    visibility = ["//third_party:__pkg__"],
)

pkg_tar(
    name = "ijar_archive",
    srcs = IJAR_SOURCES_AND_BINARIES,
    extension = "tar.gz",
    mode = "755",
    # Create a new pkg_tar to preserve the directory structure (ijar/).
    package_dir = IJAR_ARCHIVE_PACKAGE_DIR,
    visibility = ["//visibility:private"],
)
