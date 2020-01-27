# The build file for apkbuilder when it's in bazel's embedded tools repo.

licenses(["notice"])  # Apache 2.0

java_binary(
    name = "embedded_apkbuilder",
    main_class = "com.android.sdklib.build.ApkBuilderMain",
    runtime_deps = [":apkbuilder_deploy_jar"],
    visibility = ["//visibility:public"],
)

java_import(
    name = "apkbuilder_deploy_jar",
    jars = [":apkbuilder_deploy.jar"]
)
