package(default_visibility = ["//src:__subpackages__"])

java_binary(
    name = "bundlemerge",
    srcs = glob(["java/**/BundleMerge.java"]),
    main_class = "com.google.devtools.build.xcode.bundlemerge.BundleMerge",
    visibility = ["//visibility:public"],
    deps = [
        ":bundlemerge_lib",
        "//src/main/protobuf:proto_bundlemerge",
        "//src/objc_tools/plmerge:plmerge_lib",
        "//third_party:protobuf",
    ],
)

java_library(
    name = "bundlemerge_lib",
    srcs = glob(["java/**/BundleMerging.java"]),
    deps = [
        "//src/java_tools/singlejar:libSingleJar",
        "//src/java_tools/singlejar:zip",
        "//src/main/protobuf:proto_bundlemerge",
        "//src/objc_tools/plmerge:plmerge_lib",
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/common",
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/zip",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:protobuf",
        "//third_party/java/dd_plist",
    ],
)
