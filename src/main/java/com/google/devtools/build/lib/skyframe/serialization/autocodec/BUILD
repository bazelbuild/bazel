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
    exports = [":autocodec-annotation"],
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
    srcs = ["AutoCodecProcessor.java"],
    deps = [
        ":autocodec-annotation",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//third_party:auto_service",
        "//third_party:guava",
        "//third_party/java/javapoet",
        "//third_party/protobuf:protobuf_java",
    ],
)
