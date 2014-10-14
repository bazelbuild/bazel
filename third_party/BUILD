package(default_visibility = ["//visibility:public"])

filegroup(
    name = "protoc",
    srcs = select({
        ":windows_mingw": ["protobuf/protoc.exe"],
        ":windows_msys64_mingw64": ["protobuf/protoc.exe"],
        ":windows_clang": ["protobuf/protoc.exe"],
        "//conditions:default": ["protobuf/protoc.amd64"],
    }),
)

filegroup(
    name = "protobuf",
    srcs = ["protobuf/protobuf-2.5.0.jar"],
)

filegroup(
    name = "third_party",
    srcs = [
        "gson/gson-2.2.4.jar",
        "guava/guava-16.0.1.jar",
        "joda-time/joda-time-2.3.jar",
        "jsr305/jsr-305.jar",
        "protobuf/protobuf-2.5.0.jar",
    ],
)

filegroup(
    name = "third_party_test_utils",
    srcs = [
        "guava/guava-testlib.jar",
        "hamcrest/hamcrest-core-1.3.jar",
        "junit/junit-4.11.jar",
        "truth/truth-0.23.jar",
    ],
)

config_setting(
    name = "windows_mingw",
    values = { "compiler": "windows_mingw" }
)
config_setting(
    name = "windows_msys64_mingw64",
    values = { "compiler": "windows_msys64_mingw64" }
)
config_setting(
    name = "windows_clang",
    values = { "compiler": "windows_clang" }
)
