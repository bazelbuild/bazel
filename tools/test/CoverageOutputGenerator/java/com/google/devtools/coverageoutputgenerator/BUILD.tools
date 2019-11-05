package(default_visibility = ["//visibility:public"])
load("@rules_java//java:defs.bzl", "java_binary", "java_import")

java_import(
  name = "all_lcov_merger_lib",
  jars = ["@remote_coverage_tools//:all_lcov_merger_tools_deploy.jar"],
)

java_binary(
  name = "Main",
  create_executable = 1,
  main_class = "com.google.devtools.coverageoutputgenerator.Main",
  runtime_deps = [":all_lcov_merger_lib"],
)
