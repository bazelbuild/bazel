# Description:
#   Support for building swift for Apple platform

package(
    default_visibility = ["//src:__subpackages__"],
)

java_library(
    name = "swift",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:option_filters_java_proto",
        "//third_party:guava",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,  # All srcs should be not test only, overwrite package default.
    srcs = glob(["**"]),
)
