load("@rules_java//java:defs.bzl", "java_library", "java_plugin")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ),
    visibility = ["//src:__subpackages__"],
)

# @AutoCodec annotation and plugin. Used by clients. Use
# :serialization-constant instead if possible.
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

java_library(
    name = "serialization-constant",
    exported_plugins = [":serialization-constant-plugin"],
    exports = [
        ":registered-singleton",
        ":serialization-constant-annotation",
    ],
)

java_library(
    name = "serialization-constant-annotation",
    srcs = ["SerializationConstant.java"],
)

java_plugin(
    name = "serialization-constant-plugin",
    processor_class = "com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstantProcessor",
    deps = [
        ":serialization-constant-processor",
    ],
)

java_library(
    name = "serialization-processor-util",
    srcs = ["SerializationProcessorUtil.java"],
    deps = [
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party/java/javapoet",
    ],
)

java_library(
    name = "serialization-constant-processor",
    srcs = ["SerializationConstantProcessor.java"],
    deps = [
        ":registered-singleton",
        ":serialization-constant-annotation",
        ":serialization-processor-util",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:codec-scanning-constants",
        "//third_party:auto_service",
        "//third_party:guava",
        "//third_party/java/javapoet",
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
        ":serialization-constant-processor",
        ":serialization-processor-util",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/unsafe:unsafe-provider",
        "//third_party:auto_service",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party/java/javapoet",
        "//third_party/protobuf:protobuf_java",
    ],
)

# .tar archive of dependencies used for bootstrapping.
pkg_tar(
    name = "bootstrap_autocodec",
    # The .jar files are created within the .tar file under third_party/bazel_bootstrap so that
    # they will appear there in bazel-distfile, which in turn makes them visible for bootstrapping
    # from the LIBRARY_JARS rule of bazel/scripts/bootstrap/compile.sh.
    srcs = [
        ":libautocodec-annotation.jar",
        ":libautocodec-processor.jar",
        ":libregistered-singleton.jar",
        ":libserialization-constant-annotation.jar",
        ":libserialization-constant-processor.jar",
        ":libserialization-processor-util.jar",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:libserialization.jar",
        "//src/main/java/com/google/devtools/build/lib/unsafe:libunsafe-provider.jar",
    ],
    package_dir = "third_party/bazel_bootstrap",
    visibility = ["//visibility:public"],
)
