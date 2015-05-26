package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "zippingoutput",
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/singlejar:libSingleJar",
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/util",
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/zip",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
