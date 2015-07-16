java_binary(
    name = "java-bin",
    main_class = "does.not.exist",
    visibility = ["//src/tools/dash:__pkg__"],
    deps = [":servlets"],
)

java_library(
    name = "servlets",
    srcs = glob(["**/*.java"]),
    visibility = ["//src/tools/dash/src/test/java/com/google/devtools/dash:__pkg__"],
    deps = [
        "@appengine-java//:api",
        "//external:javax/servlet/api",
        "//src/main/protobuf:proto_dash",
        "//third_party:apache_velocity",
        "//third_party:guava",
    ],
)
