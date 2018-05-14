package(default_visibility = ["//visibility:public"])

java_import(
    name = "import_deps_checker_deploy_jar",
    jars = [":ImportDepsChecker_deploy.jar"]
)


java_binary(
    name = "ImportDepsChecker_embedded",
    main_class = "com.google.devtools.build.importdeps.Main",
    runtime_deps = [
        ":import_deps_checker_deploy_jar"
    ],
)