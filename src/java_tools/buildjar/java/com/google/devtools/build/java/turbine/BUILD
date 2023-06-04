load("@rules_java//java:defs.bzl", "java_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"],
)

java_binary(
    name = "turbine_direct_binary",
    main_class = "com.google.devtools.build.java.turbine.TurbineWorkerWrapper",
    srcs = ["TurbineWorkerWrapper.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/worker:work_request_handlers",
        "//third_party:turbine",
    ],
    runtime_deps = [
        "//src/main/protobuf:deps_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
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
