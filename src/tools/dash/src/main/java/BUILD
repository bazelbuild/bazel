java_binary(
    name = "java-bin",
    srcs = glob(["**/*.java"]),
    main_class = "does.not.exist",
    visibility = ["//src/tools/dash:__pkg__"],
    deps = [
        "@appengine-java//:api",
        "//external:javax/servlet/api",
        "//src/main/protobuf:proto_dash",
        "//third_party:apache_velocity",
        "//third_party:guava",
    ],
)
