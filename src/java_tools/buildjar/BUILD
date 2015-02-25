package(default_visibility = ["//src:__pkg__"])

java_binary(
    name = "JavaBuilder",
    srcs = glob(["java/**/*.java"]),
    main_class = "com.google.devtools.build.buildjar.BazelJavaBuilder",
    deps = [
        "//src/main/protobuf:proto_deps",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:protobuf",
        "//tools/jdk:langtools-neverlink",
    ],
)
