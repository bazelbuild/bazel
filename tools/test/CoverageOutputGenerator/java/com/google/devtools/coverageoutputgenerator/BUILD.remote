package(default_visibility = ["//visibility:public"])

load("@rules_java//java:java_binary.bzl", "java_binary")
load("@rules_java//java:java_import.bzl", "java_import")

java_import(
    name = "all_lcov_merger_lib",
    jars = ["all_lcov_merger_tools_deploy.jar"],
)

java_binary(
    name = "Main",
    create_executable = 1,
    main_class = "com.google.devtools.coverageoutputgenerator.Main",
    runtime_deps = [":all_lcov_merger_lib"],
)

alias(
    name = "lcov_merger",
    actual = "Main",
)

alias(
    name = "coverage_report_generator",
    actual = "Main",
)
