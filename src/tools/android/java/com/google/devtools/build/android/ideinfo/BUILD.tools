package(default_visibility = ["//visibility:public"])

java_import(
    name = "classes",
    jars = [":classes_deploy.jar"],
)

java_binary(
    name = "PackageParser",
    main_class = "com.google.devtools.build.android.ideinfo.PackageParser",
    runtime_deps = [":classes"],
)
