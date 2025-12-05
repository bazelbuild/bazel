load("@rules_java//java:defs.bzl", "java_binary", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/build/stack/devtools/build/constellate/fakebuildapi:srcs",
        "//src/main/java/build/stack/devtools/build/constellate/renderer:srcs",
        "//src/main/java/build/stack/devtools/build/constellate/rendering:srcs",
        "//src/main/java/build/stack/devtools/build/constellate/rendering/proto:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "constellate_lib",
    srcs = [
        "Constellate.java",
        "FilesystemFileAccessor.java",
        "StarlarkFileAccessor.java",
        "SystemOutEventHandler.java",
    ],
    deps = [
        # "//src/main/java/build/stack/devtools/build/constellate/fakebuildapi",
        "//src/main/java/build/stack/devtools/build/constellate/rendering",
        "//src/main/java/build/stack/starlark/v1beta1:starlark_java_grpc",
        "//src/main/java/build/stack/starlark/v1beta1:starlark_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/protobuf:stardoc_output_java_proto",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/lib/json",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:flogger",
        "//third_party:guava",
    ],
)

java_library(
    name = "server_lib",
    srcs = [
        "RemoteStarlarkOptions.java",
        "RemoteStarlarkServer.java",
        "StarlarkServer.java",
        "StatusUtils.java",
    ],
    deps = [
        ":constellate_lib",
        "//src/main/java/build/stack/devtools/build/constellate/rendering",
        "//src/main/java/build/stack/starlark/v1beta1:starlark_java_grpc",
        "//src/main/java/build/stack/starlark/v1beta1:starlark_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/cmdline:LabelValidator",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/lib/util:single_line_formatter",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skydoc/rendering/proto:stardoc_output_java_proto",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:netty",
        "//third_party/grpc:grpc-jar",
        "//third_party/protobuf:protobuf_java",
        "//third_party/protobuf:protobuf_java_util",
        "@googleapis//:google_rpc_code_java_proto",
        "@googleapis//:google_rpc_error_details_java_proto",
        "@googleapis//:google_rpc_status_java_proto",
    ],
)

java_binary(
    name = "server",
    jvm_flags = [
        # quiet warnings from com.google.protobuf.UnsafeUtil,
        # see: https://github.com/google/protobuf/issues/3781
        # and: https://github.com/bazelbuild/bazel/issues/5599
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        # ... but only on JDK >= 9
        "-XX:+IgnoreUnrecognizedVMOptions",
    ],
    main_class = "build.stack.devtools.build.constellate.RemoteStarlarkServer",
    runtime_deps = [
        ":server_lib",
    ],
)
