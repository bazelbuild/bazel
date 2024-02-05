load("@rules_java//java:defs.bzl", "java_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__pkg__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/execlog/src/main/java/com/google/devtools/build/execlog:srcs",
        "//src/tools/execlog/test/main/java/com/google/devtools/build/execlog:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_binary(
    name = "parser",
    main_class = "com.google.devtools.build.execlog.ExecLogParser",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/tools/execlog/src/main/java/com/google/devtools/build/execlog:parser"],
)

java_binary(
    name = "converter",
    main_class = "com.google.devtools.build.execlog.ExecLogConverter",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/tools/execlog/src/main/java/com/google/devtools/build/execlog:converter"],
)
