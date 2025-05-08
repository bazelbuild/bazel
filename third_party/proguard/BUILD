load("@rules_java//java:java_binary.bzl", "java_binary")
load("@rules_java//java:java_import.bzl", "java_import")

java_binary(
    name = "proguard",
    main_class = "proguard.ProGuard",
    visibility = ["//visibility:public"],
    runtime_deps = [":proguard_lib"],
)

java_import(
    name = "proguard_lib",
    jars = ["lib/proguard.jar"],
)
