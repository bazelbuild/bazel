# Description:
#   Support for building cpp for Apple platform

package(
    default_visibility = ["//src:__subpackages__"],
)

java_library(
    name = "cpp",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:common",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib/rules/apple",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/protobuf:xcodegen_java_proto",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
