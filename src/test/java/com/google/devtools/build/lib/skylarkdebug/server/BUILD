package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "utils",
    srcs = ["MockDebugClient.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/proto:skylark_debugging_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_test(
    name = "SkylarkDebugServerTest",
    size = "medium",
    srcs = ["SkylarkDebugServerTest.java"],
    deps = [
        ":utils",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/proto:skylark_debugging_java_proto",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/server",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib:foundations_testutil",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DebugServerTransportTest",
    srcs = ["DebugServerTransportTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/proto:skylark_debugging_java_proto",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/server",
        "//src/test/java/com/google/devtools/build/lib:foundations_testutil",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DebuggerSerializationTest",
    srcs = glob(["DebuggerSerializationTest.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/proto:skylark_debugging_java_proto",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/server",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
