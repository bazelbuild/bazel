package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load("@bazel_tools//third_party/protobuf:protobuf.bzl", "py_proto_library")

filegroup(
    name = "srcs",
    srcs = glob(
        [
            "**",
            "protobuf.bzl",
            "python/google/protobuf/**/*.py",
            "src/google/protobuf/**/*.proto",
        ],
    ),
    visibility = ["//third_party:__pkg__"],
)

filegroup(
    name = "protoc",
    srcs = select({
        "//third_party:windows_mingw": ["protoc-windows-x86_64.exe"],
        "//third_party:windows_msys64_mingw64": ["protoc-windows-x86_64.exe"],
        "//third_party:windows_msys64": ["protoc-windows-x86_64.exe"],
        "//third_party:windows_clang": ["protoc-windows-x86_64.exe"],
        "//third_party:darwin": ["protoc-osx-x86_32.exe"],
        "//third_party:k8": ["protoc-linux-x86_64.exe"],
        "//third_party:piii": ["protoc-linux-x86_32.exe"],
        "//third_party:freebsd": ["protoc-linux-x86_32.exe"],
    }),
)

java_import(
    name = "protobuf",
    jars = ["protobuf-java-3.0.0-beta-1.jar"],
)

# For bootstrapping JavaBuilder
filegroup(
    name = "protobuf-jars",
    srcs = ["protobuf-java-3.0.0-beta-1.jar"],
)

py_library(
    name = "protobuf_python_srcs",
    srcs = glob(
        [
            "python/google/protobuf/*.py",
            "python/google/protobuf/**/*.py",
        ],
    ),
    imports = ["python"],
)

py_proto_library(
    name = "protobuf_python",
    srcs = [
        "src/google/protobuf/any.proto",
        "src/google/protobuf/api.proto",
        "src/google/protobuf/compiler/plugin.proto",
        "src/google/protobuf/descriptor.proto",
        "src/google/protobuf/duration.proto",
        "src/google/protobuf/empty.proto",
        "src/google/protobuf/field_mask.proto",
        "src/google/protobuf/source_context.proto",
        "src/google/protobuf/struct.proto",
        "src/google/protobuf/timestamp.proto",
        "src/google/protobuf/type.proto",
        "src/google/protobuf/wrappers.proto",
    ],
    include = "src",
    default_runtime = "",
    protoc = ":protoc",
    py_libs = [
        ":protobuf_python_srcs",
        "//third_party/py/six",
    ],
    srcs_version = "PY2AND3",
)
