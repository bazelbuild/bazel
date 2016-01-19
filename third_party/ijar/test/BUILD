package(default_visibility = ["//visibility:__pkg__"])

licenses(["notice"])  # Apache License 2.0

exports_files(["LICENSE"])

sh_test(
    name = "ijar_test",
    size = "enormous",
    srcs = ["ijar_test.sh"],
    args = [
        "$(JAVAC)",
        "$(JAVA)",
        "$(JAVABASE)/bin/jar",
        "$(JAVABASE)/bin/javap",
        "$(location //third_party/ijar)",
        "$(location //tools/jdk:langtools)",
        # We assume unzip and zip to be on the path
        "unzip",
        "zip",
        "$(location :zip_count)",
    ],
    data = [
        "testenv.sh",
        "//src/test/shell:bashunit",
        "//third_party/ijar",
        "A.java",
        "B.java",
        "Object.java",
        "WellCompressed1.java",
        "WellCompressed2.java",
        ":libtypeannotations2.jar",
        ":libmethodparameters.jar",
        ":source_debug_extension.jar",
        ":largest_regular.jar",
        ":smallest_zip64.jar",
        ":definitely_zip64.jar",
        ":zip_count",
        "TypeAnnotationTest2.java",
        # invokedynamic/ClassWithLambda.java, compiled with javac8
        ":libinvokedynamic.jar",
        # Intentionally bad test JAR: file count in central directory
        # wrong.
        "libwrongcentraldir.jar",
        "//tools/defaults:jdk",
        "//tools/jdk:langtools",
    ],
    shard_count = 5,
    tags = ["zip"],
)

sh_test(
    name = "zip_test",
    size = "large",
    srcs = ["zip_test.sh"],
    args = [
        "$(location //third_party/ijar:zipper)",
        # We assume unzip and zip to be on the path
        "unzip",
        "zip",
    ],
    data = [
        "testenv.sh",
        "//src/test/shell:bashunit",
        "//third_party/ijar:zipper",
    ],
    tags = ["zip"],
)

java_library(
    name = "invokedynamic",
    testonly = 1,
    srcs = ["invokedynamic/ClassWithLambda.java"],
    javacopts = ["-source 8 -target 8"],
    tags = ["jdk8"],
)

# For testing purposes: a class which exposes a private nested class in
# public API, classes with annotations and a class with different forms of
# deprecations markers
java_library(
    name = "ijar_testlib",
    srcs = [
        "AnnotatedClass.java",
        "Annotations.java",
        "DeprecatedParts.java",
        "DumbClass.java",
        "EnclosingMethod.java",
        "PrivateMembersClass.java",
    ],
)

# For testing purposes: an interface jar (ijar) which includes the
# interface view of the various test class from above.
genrule(
    name = "interface_ijar_testlib",
    srcs = [":ijar_testlib"],
    outs = ["interface_ijar_testlib.jar"],
    cmd = "$(location //third_party/ijar) $< $@",
    tools = ["//third_party/ijar"],
)

java_library(
    name = "typeannotations2",
    srcs = glob(["typeannotations2/**/*.java"]),
    javacopts = ["-source 8 -target 8"],
    tags = ["jdk8"],
)

java_library(
    name = "methodparameters",
    srcs = ["methodparameters/Test.java"],
    javacopts = ["-source 8 -target 8 -parameters"],
)

java_binary(
    name = "gen_source_debug_extension",
    srcs = ["GenSourceDebugExtension.java"],
    main_class = "test.GenSourceDebugExtension",
    deps = ["//third_party:asm"],
)

genrule(
    name = "gen_source_debug_extension_jar",
    srcs = [],
    outs = ["source_debug_extension.jar"],
    cmd = "$(location :gen_source_debug_extension) $@",
    tools = ["gen_source_debug_extension"],
)

java_test(
    name = "IjarTests",
    size = "small",
    srcs = [
        "IjarTests.java",
        "StripVerifyingVisitor.java",
    ],
    data = [
        "PrivateNestedClass.java",
        "UseDeprecatedParts.java",
        "UseRestrictedAnnotation.java",
        "package-info.java",
        ":interface_ijar_testlib",
    ],
    tags = ["zip"],
    test_class = "IjarTests",
    deps = [
        "//src/java_tools/buildjar:BazelJavaCompiler",
        "//third_party:asm",
        "//third_party:junit4",
    ],
)

test_suite(
    name = "all_tests",
    testonly = 1,
    visibility = ["//visibility:public"],
)

java_binary(
    name = "gen_zip_with_entries",
    srcs = ["GenZipWithEntries.java"],
    jvm_flags = ["-Djdk.util.zip.inhibitZip64=false"],
    main_class = "test.GenZipWithEntries",
    deps = ["//third_party:asm"],
)

[
    genrule(
        name = name,
        outs = [name + ".jar"],
        cmd = "$(location :gen_zip_with_entries) %s $@" % entries,
        tools = [":gen_zip_with_entries"],
    )
    for name, entries in [
        ("largest_regular", 65535),
        ("smallest_zip64", 65536),
        ("definitely_zip64", 70000),
    ]
]

java_binary(
    name = "zip_count",
    srcs = ["ZipCount.java"],
    jvm_flags = ["-Djdk.util.zip.inhibitZip64=false"],
    main_class = "test.ZipCount",
)
