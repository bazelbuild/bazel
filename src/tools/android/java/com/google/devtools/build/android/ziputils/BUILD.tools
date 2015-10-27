package(default_visibility = ["//visibility:public"])

java_import(
    name = "classes",
    jars = [":all_deploy.jar"],
)

java_binary(
    name = "mapper",
    main_class = "com.google.devtools.build.android.ziputils.DexMapper",
    runtime_deps = [":classes"],
)

java_binary(
    name = "reducer",
    main_class = "com.google.devtools.build.android.ziputils.DexReducer",
    runtime_deps = [":classes"],
)
