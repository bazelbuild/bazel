package(default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"])

java_binary(
    name = "turbine",
    main_class = "com.google.devtools.build.java.turbine.Turbine",
    visibility = [
        "//:__subpackages__",
        "//src/java_tools/buildjar:buildjar_package_group",
    ],
    runtime_deps = [":turbine_main"],
)

java_library(
    name = "turbine_main",
    srcs = ["Turbine.java"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:javac_turbine",
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
