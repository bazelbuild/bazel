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
        ":registered-singleton",
        "//third_party/protobuf:protobuf_java",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/unsafe:unsafe-provider",
    ],
)

java_library(
    name = "registered-singleton",
    srcs = ["RegisteredSingletonDoNotUse.java"],
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

# @AutoCodec annotation processor implementation.
java_library(
    name = "autocodec-processor",
    srcs = [
        "AutoCodecProcessor.java",
        "AutoCodecUtil.java",
        "Marshallers.java",
        "SerializationCodeGenerator.java",
    ],
    deps = [
        ":autocodec-annotation",
        ":registered-singleton",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/unsafe:unsafe-provider",
        "//third_party:auto_service",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party/java/javapoet",
        "//third_party/protobuf:protobuf_java",
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
        ":libautocodec-processor.jar": "third_party/bazel_bootstrap/libautocodec-processor.jar",
        ":libregistered-singleton.jar": "third_party/bazel_bootstrap/libregistered-singleton.jar",
        "//src/main/java/com/google/devtools/build/lib/unsafe:libunsafe-provider.jar": "third_party/bazel_bootstrap/libunsafe-provider.jar",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:libserialization.jar": "third_party/bazel_bootstrap/libserialization.jar",
    },
    visibility = ["//visibility:public"],
)
