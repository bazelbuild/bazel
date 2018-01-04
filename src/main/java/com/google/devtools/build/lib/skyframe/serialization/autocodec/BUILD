package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ),
)

# @AutoCodec annotation and plugin. Used by clients.
java_library(
    name = "autocodec",
    exported_plugins = [":autocodec-plugin"],
    exports = [
        ":autocodec-annotation",
        # Generated classes have the following dependencies.
        ":unsafe-provider",
        "//third_party/protobuf:protobuf_java",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
    ],
)

# @AutoCodec annotation only. Used by clients and the processor.
java_library(
    name = "autocodec-annotation",
    srcs = ["AutoCodec.java"],
)

# Installs the @AutoCodec annotation processor as a compiler plugin.
java_plugin(
    name = "autocodec-plugin",
    processor_class = "com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodecProcessor",
    deps = [
        ":autocodec-processor",
    ],
)

java_library(
    name = "unsafe-provider",
    srcs = ["UnsafeProvider.java"],
)

# @AutoCodec annotation processor implementation.
java_library(
    name = "autocodec-processor",
    srcs = [
        "AutoCodecProcessor.java",
        "Marshaller.java",
        "Marshallers.java",
    ],
    deps = [
        ":autocodec-annotation",
        ":autocodec-util",
        ":unsafe-provider",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//third_party:auto_service",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/javapoet",
        "//third_party/protobuf:protobuf_java",
    ],
)

# Code shared between autocodec-processor and fake-autocodec-processor.
java_library(
    name = "autocodec-util",
    srcs = ["AutoCodecUtil.java"],
    deps = [
        ":autocodec-annotation",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/javapoet",
        "//third_party/protobuf:protobuf_java",
    ],
)

# This is a fake implementation for bootstrapping. It must *not* be used in production.
java_library(
    name = "fake-autocodec-processor",
    srcs = ["FakeAutoCodecProcessor.java"],
    deps = [
        ":autocodec-util",
        "//third_party:auto_service",
        "//third_party:guava",
        "//third_party/java/javapoet",
    ],
)

load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

# .tar archive of dependencies used for bootstrapping.
pkg_tar(
    name = "bootstrap_autocodec",
    # The .jar files are created within the .tar file under third_party/bazel_bootstrap so that
    # they will appear there in bazel-distfile, which in turn makes them visible for bootstrapping
    # from the LIBRARY_JARS rule of bazel/scripts/bootstrap/compile.sh.
    files = {
        ":libautocodec-annotation.jar": "third_party/bazel_bootstrap/libautocodec-annotation.jar",
        ":libautocodec-util.jar": "third_party/bazel_bootstrap/libautocodec-util.jar",
        ":libfake-autocodec-processor.jar": "third_party/bazel_bootstrap/libfake-autocodec-processor.jar",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:libserialization.jar": "third_party/bazel_bootstrap/libserialization.jar",
    },
    visibility = ["//visibility:public"],
)
