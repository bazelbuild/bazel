load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_test(
    name = "VanillaJavaBuilderTest",
    srcs = ["VanillaJavaBuilderTest.java"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:vanilla_java_builder",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:io",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "JavacWorkDirStrippingTest",
    srcs = ["JavacWorkDirStrippingTest.java"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:processing",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)
