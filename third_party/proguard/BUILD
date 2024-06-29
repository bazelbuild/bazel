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
