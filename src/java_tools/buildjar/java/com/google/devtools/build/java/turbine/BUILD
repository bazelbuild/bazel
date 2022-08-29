load("@rules_java//java:defs.bzl", "java_binary")

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

java_library(
    name = "turbine_wrapper",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/worker:work_request_handlers",
        "//third_party:turbine",
    ],
)

java_binary(
    name = "turbine_binary",
    main_class = "com.google.devtools.build.java.turbine.BazelTurbine",
    runtime_deps = [":turbine_wrapper"],
)

filegroup(
    name = "srcs",
    srcs = glob(
        ["**/*.java"],
        allow_empty = True,
    ) + [
        "BUILD",
    ],
    visibility = ["//src:__subpackages__"],
)
