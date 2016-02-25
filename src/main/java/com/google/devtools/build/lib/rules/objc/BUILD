# Description:
#   Objective-C rule support

package(
    default_visibility = ["//src:__subpackages__"],
)

java_library(
    name = "objc",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:build-info",
        "//src/main/java/com/google/devtools/build/lib:collect",
        "//src/main/java/com/google/devtools/build/lib:common",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:java-rules",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib:shell",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/rules/apple",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:bundlemerge_java_proto",
        "//src/main/protobuf:plmerge_java_proto",
        "//src/main/protobuf:xcodegen_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/dd_plist",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
