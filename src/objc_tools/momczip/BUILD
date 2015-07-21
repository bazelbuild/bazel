package(default_visibility = ["//visibility:public"])

java_binary(
    name = "momczip",
    srcs = glob(["java/**/*.java"]),
    main_class = "com.google.devtools.build.xcode.momczip.MomcZip",
    deps = [
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/zippingoutput",
        "//third_party:guava",
    ],
)
