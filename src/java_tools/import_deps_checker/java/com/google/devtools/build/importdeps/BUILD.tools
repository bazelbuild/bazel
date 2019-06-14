package(default_visibility = ["//visibility:public"])

java_import(
    name = "import_deps_checker_deploy_jar",
    jars = ["@android_tools//:ImportDepsChecker_deploy.jar"],
)

java_binary(
    name = "ImportDepsChecker_embedded",
    jvm_flags = [
        # quiet warnings from com.google.protobuf.UnsafeUtil,
        # see: https://github.com/google/protobuf/issues/3781
        "-XX:+IgnoreUnrecognizedVMOptions",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
    ],
    main_class = "com.google.devtools.build.importdeps.Main",
    runtime_deps = [
        ":import_deps_checker_deploy_jar",
    ],
)
