package(default_visibility = ["//visibility:public"])

java_import(
    name = "all_android_tools",
    jars = ["@android_tools//:all_android_tools_deploy.jar"],
)

java_binary(
    name = "ResourceProcessorBusyBox",
    jvm_flags = [
        # quiet warnings from com.google.protobuf.UnsafeUtil,
        # see: https://github.com/google/protobuf/issues/3781
        "-XX:+IgnoreUnrecognizedVMOptions",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
    ],
    main_class = "com.google.devtools.build.android.ResourceProcessorBusyBox",
    runtime_deps = [
        ":all_android_tools",
    ],
)

java_binary(
    name = "ZipFilterAction",
    # Memory consumption of SingleJar is about 250 bytes per entry in the output file. Unfortunately,
    # the JVM tends to kill the process with an OOM long before we're at the limit. In the most
    # recent example, 400 MB of memory was enough for about 500,000 entries.
    jvm_flags = ["-Xmx1600m"],
    main_class = "com.google.devtools.build.android.ZipFilterAction",
    runtime_deps = [
        ":all_android_tools",
    ],
)
