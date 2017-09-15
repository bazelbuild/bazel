package(default_visibility = ["//src:__subpackages__"])

java_binary(
    name = "turbine",
    main_class = "com.google.devtools.build.java.turbine.Turbine",
    runtime_deps = [":turbine_main"],
)

java_library(
    name = "turbine_main",
    srcs = ["Turbine.java"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:javac_turbine",
        "//third_party:guava",
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
