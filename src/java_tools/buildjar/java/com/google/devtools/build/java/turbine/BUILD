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
    name = "turbine_direct",
    # TODO(cushon): consolidate with select in //third_party:turbine_direct.
    srcs = select({
        "//src/conditions:darwin": ["//third_party:turbine_direct"],
        "//src/conditions:darwin_x86_64": ["//third_party:turbine_direct"],
        "//src/conditions:linux_x86_64": ["//third_party:turbine_direct"],
        "//conditions:default": ["turbine_direct_binary_deploy.jar"],
    }),
    visibility = [
        "//:__subpackages__",
    ],
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
