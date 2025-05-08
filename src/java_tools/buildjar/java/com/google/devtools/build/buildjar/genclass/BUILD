load("@rules_java//java:defs.bzl", "java_binary", "java_library")
load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*.java"]) + ["BUILD"],
    visibility = ["//src:__subpackages__"],
)

java_binary(
    name = "GenClass",
    main_class = "com.google.devtools.build.buildjar.genclass.GenClass",
    visibility = [
        "//src:__pkg__",
        "//src/java_tools/buildjar:buildjar_package_group",
        "//src/test/shell/bazel:__pkg__",
    ],
    runtime_deps = [":genclass_lib"],
)

java_library(
    name = "genclass_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:jarhelper",
        "//src/main/protobuf:java_compilation_java_proto",
        "//third_party:guava",
    ],
)

#
## Bootstrapping using Starlark rules
#

bootstrap_java_binary(
    name = "bootstrap_genclass",
    srcs = glob(["*.java"]),
    main_class = "com.google.devtools.build.buildjar.genclass.GenClass",
    tags = ["manual"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:starlark-deps",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper:bootstrap_jarhelper",
    ],
)
