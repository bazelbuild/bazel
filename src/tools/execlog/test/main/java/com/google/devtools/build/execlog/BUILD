package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(
        ["**"],
        exclude = [
            "*~",
        ],
    ),
    visibility = ["//src/tools/execlog:__pkg__"],
)

java_test(
    name = "ExecLogParserTest",
    size = "small",
    srcs = ["ExecLogParserTest.java"],
    test_class = "com.google.devtools.build.execlog.ExecLogParserTest",
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:spawn_java_proto",
        "//src/tools/execlog/src/main/java/com/google/devtools/build/execlog:parser",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
