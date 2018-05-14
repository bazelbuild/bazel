package(default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_test(
    name = "VanillaJavaBuilderTest",
    srcs = ["VanillaJavaBuilderTest.java"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:vanilla_java_builder",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
