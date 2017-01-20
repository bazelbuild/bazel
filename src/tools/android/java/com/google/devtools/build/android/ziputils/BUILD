# Low level zip archive processing library.

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        ":libsplitter_lib.jar",
        ":libziputils_lib.jar",
        ":mapper_bin.jar",
        ":reducer_bin.jar",
    ],
)

java_library(
    name = "ziputils_lib",
    srcs = glob(
        ["*.java"],
        exclude = [
            "DexMapper.java",
            "DexReducer.java",
            "SplitZip.java",
            "SplitZipFilters.java",
            "Splitter.java",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "splitter_lib",
    srcs = [
        "SplitZip.java",
        "SplitZipFilters.java",
        "Splitter.java",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":ziputils_lib",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)


java_binary(
    name = "mapper_bin",
    srcs = [
        "DexMapper.java",
    ],
    main_class = "com.google.devtools.build.android.ziputils.DexMapper",
    visibility = ["//visibility:public"],
    deps = [
        ":splitter_lib",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_binary(
    name = "reducer_bin",
    srcs = [
        "DexReducer.java",
    ],
    main_class = "com.google.devtools.build.android.ziputils.DexReducer",
    visibility = ["//visibility:public"],
    deps = [
        ":ziputils_lib",
        "//src/main/java/com/google/devtools/common/options",
    ],
)
