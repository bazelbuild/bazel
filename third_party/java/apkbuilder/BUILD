licenses(["notice"])  # Apache 2.0

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "apkbuilder_deploy.jar",
    ],
    visibility = ["//src:__pkg__"],
)

java_binary(
    name = "apkbuilder",
    srcs = glob(["java/**/*.java"]),
    main_class = "com.android.sdklib.build.ApkBuilderMain",
    deps = [
        "//src/java_tools/singlejar:libSingleJar",
        "//third_party:guava",
    ],
)
