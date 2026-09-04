load("@rules_java//java:defs.bzl", "java_library", "java_plugin")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ),
    visibility = ["//src/main/java/com/google/devtools/build/lib/skyframe/serialization:__pkg__"],
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
        ":runtime-helpers",
        "//third_party/java/guava:collect",
        "//third_party/java/protobuf",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/unsafe:unsafe-provider",
    ],
)

java_library(
    name = "runtime-helpers",
    srcs = ["RuntimeHelpers.java"],
)

java_library(
    name = "registered-singleton",
    srcs = ["RegisteredSingletonDoNotUse.java"],
    visibility = ["//src/main/java/com/google/devtools/build/lib/skyframe/serialization:__pkg__"],
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
    visibility = ["//visibility:private"],
)

java_plugin(
    name = "serialization-constant-plugin",
    processor_class = "com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstantProcessor",
    visibility = ["//visibility:private"],
    deps = [
        ":serialization-constant-processor",
    ],
)

java_library(
    name = "serialization-processing-exception",
    srcs = ["SerializationProcessingException.java"],
    visibility = ["//visibility:private"],
    deps = ["//third_party/java/error_prone:annotations"],
)

java_library(
    name = "type-operations",
    srcs = ["TypeOperations.java"],
    visibility = ["//visibility:private"],
    deps = [
        ":serialization-processing-exception",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/javapoet",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "serialization-constant-processor",
    srcs = ["SerializationConstantProcessor.java"],
    visibility = ["//visibility:private"],
    deps = [
        ":registered-singleton",
        ":serialization-constant-annotation",
        ":serialization-processing-exception",
        ":type-operations",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:codec-scanning-constants",
        "//third_party/java/auto:auto_service",
        "//third_party/java/guava:collect",
        "//third_party/java/javapoet",
    ],
)

# @AutoCodec annotation only. Clients should not depend on this directly as it doesn't include the processor.
# Depend on :autocodec instead.
java_library(
    name = "autocodec-annotation",
    srcs = ["AutoCodec.java"],
    visibility = ["//visibility:private"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
    ],
)

# Installs the @AutoCodec annotation processor as a compiler plugin.
java_plugin(
    name = "autocodec-plugin",
    processor_class = "com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodecProcessor",
    visibility = ["//visibility:private"],
    deps = [
        ":autocodec-processor",
    ],
)

# @AutoCodec annotation processor implementation.
java_library(
    name = "autocodec-processor",
    srcs = [
        "AutoCodecAnnotation.java",
        "AutoCodecProcessor.java",
        "CodecGenerator.java",
        "DeferredObjectCodecConstants.java",
        "DeferredObjectCodecFieldGenerators.java",
        "DeferredObjectCodecGenerator.java",
        "FieldGenerator.java",
        "Initializers.java",
        "InterningObjectCodecFieldGenerators.java",
        "InterningObjectCodecGenerator.java",
        "SerializationCodeGenerator.java",
    ],
    deps = [
        ":autocodec-annotation",
        ":runtime-helpers",
        ":serialization-processing-exception",
        ":type-operations",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/unsafe:unsafe-provider",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//third_party/java/auto:auto_service",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/javapoet",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/protobuf:protobuf_lite",
    ],
)
