package(default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"])

java_binary(
    name = "turbine_direct_binary",
    main_class = "com.google.turbine.main.Main",
    runtime_deps = [
        "//src/main/protobuf:deps_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:turbine",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + [
        "BUILD",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:srcs",
    ],
)
