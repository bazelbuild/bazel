package(default_visibility = ["//visibility:public"])

java_import(
    name = "classes",
    jars = [":classes_deploy.jar"],
)

java_binary(
    name = "IdlClass",
    main_class = "com.google.devtools.build.android.idlclass.IdlClass",
    runtime_deps = [":classes"],
)
