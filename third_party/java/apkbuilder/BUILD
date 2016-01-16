licenses(["notice"])  # Apache 2.0

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "apkbuilder_deploy.jar",
    ],
    visibility = ["//src:__pkg__"],
)

# This filegroup is necessary so that Bazel finds an apkbuilder during the
# bootstrap process when this file is used as
# @bazel_tools//third_party/java/apkbuilder/BUILD. Otherwise, the bootstrap
# would fail if an Android SDK was declared in the WORKSPACE file of Bazel.
filegroup(name = "embedded_apkbuilder")

java_binary(
    name = "apkbuilder",
    srcs = glob(["java/**/*.java"]),
    main_class = "com.android.sdklib.build.ApkBuilderMain",
    deps = [
        "//src/java_tools/singlejar:libSingleJar",
        "//third_party:guava",
    ],
)
