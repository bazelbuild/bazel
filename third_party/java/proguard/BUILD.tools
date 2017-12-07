java_import(
    name = "proguard_import",
    jars = ["proguard5.3.3/lib/proguard.jar"],
)

java_binary(
    name = "proguard",
    main_class = "proguard.ProGuard",
    runtime_deps = [":proguard_import"],
    visibility = ["//visibility:public"],
)
