package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_binary(
    name = "swiftstdlibtoolzip",
    srcs = ["SwiftStdlibToolZip.java"],
    main_class = "com.google.devtools.build.xcode.swiftstdlibtoolzip.SwiftStdlibToolZip",
    visibility = ["//visibility:public"],
    deps = [
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/zippingoutput",
        "//third_party:guava",
    ],
)
