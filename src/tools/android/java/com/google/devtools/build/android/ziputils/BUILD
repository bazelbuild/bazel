# Low level zip archive processing library.

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "ziputils_lib",
    srcs = glob(
        ["*.java"],
        exclude = [
            "DexMapper.java",
            "DexReducer.java",
            "SplitZip.java",
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
    srcs = glob(
        [
            "SplitZip.java",
            "Splitter.java",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":ziputils_lib",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_binary(
    name = "mapper",
    srcs = [
        "DexMapper.java",
    ],
    main_class = "com.google.devtools.build.android.ziputils.DexMapper",
    visibility = ["//visibility:public"],
    deps = [
        ":splitter_lib",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_binary(
    name = "reducer",
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
