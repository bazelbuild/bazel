package(default_visibility = ["//src:__subpackages__"])

java_binary(
    name = "plmerge",
    srcs = glob(["java/**/PlMerge.java"]),
    main_class = "com.google.devtools.build.xcode.plmerge.PlMerge",
    visibility = ["//visibility:public"],
    deps = [
        ":plmerge_lib",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:plmerge_java_proto",
        "//third_party:guava",
        "//third_party/java/dd_plist",
    ],
)

java_library(
    name = "plmerge_lib",
    srcs = glob(
        ["java/**/*.java"],
    ),
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:plmerge_java_proto",
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/common",
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/util",
        "//third_party:guava",
        "//third_party/java/dd_plist",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)
