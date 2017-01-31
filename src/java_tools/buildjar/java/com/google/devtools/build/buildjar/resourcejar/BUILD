filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:__pkg__"],
)

java_binary(
    name = "ResourceJarBuilder",
    main_class = "com.google.devtools.build.buildjar.resourcejar.ResourceJarBuilder",
    runtime_deps = [":resourcejar"],
)

java_library(
    name = "resourcejar",
    srcs = [
        "ResourceJarBuilder.java",
        "ResourceJarOptions.java",
        "ResourceJarOptionsParser.java",
    ],
    visibility = [
        "//src/java_tools/buildjar/javatests/com/google/devtools/build/buildjar/resourcejar:__pkg__",
    ],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper",
        "//third_party:guava",
    ],
)
