package_group(
    name = "packages",
    packages = [
        "//src/java_tools/buildjar/...",
    ],
)

package(default_visibility = [":packages"])

java_binary(
    name = "GenClass",
    main_class = "com.google.devtools.build.buildjar.genclass.GenClass",
    visibility = ["//visibility:public"],
    runtime_deps = [":genclass_lib"],
)

java_library(
    name = "genclass_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper",
        "//src/main/protobuf:java_compilation_java_proto",
        "//third_party:guava",
    ],
)

#
## Bootstrapping using Skylark rules
#

load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library", "bootstrap_java_binary")

bootstrap_java_binary(
    name = "bootstrap_genclass",
    srcs = glob(["*.java"]),
    main_class = "com.google.devtools.build.buildjar.genclass.GenClass",
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:skylark-deps",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper:bootstrap_jarhelper",
    ],
)
