package(default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"])

java_test(
    name = "TurbineTest",
    srcs = ["TurbineTest.java"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/bazel:JavacBootclasspath",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:turbine_main",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party:turbine",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["*.java"]) + [
        "BUILD",
        "//src/java_tools/buildjar/javatests/com/google/devtools/build/java/turbine/javac:srcs",
    ],
)
