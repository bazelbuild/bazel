load("@rules_java//java:defs.bzl", "java_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__pkg__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/execgraph/src/main/java/com/google/devtools/build/execgraph:srcs",
        "//src/tools/execgraph/test/main/java/com/google/devtools/build/execgraph:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_binary(
    name = "parser",
    main_class = "com.google.devtools.build.execgraph.ExecGraphParser",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/tools/execgraph/src/main/java/com/google/devtools/build/execgraph:parser"],
)
