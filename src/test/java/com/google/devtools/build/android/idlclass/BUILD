filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/android:__pkg__"],
)

java_test(
    name = "IdlClassTest",
    size = "small",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/protobuf:java_compilation_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android/idlclass:idlclass_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
