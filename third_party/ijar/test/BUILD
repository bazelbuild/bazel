package(default_visibility = ["//visibility:__pkg__"])

licenses(["notice"])  # Apache License 2.0

exports_files(["LICENSE"])

sh_test(
    name = "ijar_test",
    size = "enormous",
    srcs = ["ijar_test.sh"],
    args = [
        "../local_jdk/bin/javac",
        "../local_jdk/bin/java",
        "../local_jdk/bin/jar",
        "../local_jdk/bin/javap",
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
        ":libinvokedynamic.jar",
        # Intentionally bad test JAR: file count in central directory
        # wrong.
        "libwrongcentraldir.jar",
        "//tools/defaults:jdk",
        "//tools/jdk:langtools",
    ],
    shard_count = 5,
    tags = [
        "jdk8",
        "zip",
    ],
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

genrule(
    name = "interface_ijar_testlib_with_target_label",
    srcs = [":ijar_testlib"],
    outs = ["interface_ijar_testlib_with_target_label.jar"],
    cmd = "$(location //third_party/ijar) --target_label //foo:foo --injecting_rule_kind foo_library $< $@",
    tools = ["//third_party/ijar"],
)

genrule(
    name = "interface_ijar_testlib_nostrip",
    srcs = [":ijar_testlib"],
    outs = ["interface_ijar_testlib_nostrip.jar"],
    cmd = "$(location //third_party/ijar) --target_label //foo:foo --nostrip_jar $< $@",
    tools = ["//third_party/ijar"],
)

genrule(
    name = "jar_with_manifest_nostrip",
    srcs = ["jar-with-manifest.jar"],
    outs = ["jar-with-manifest-nostrip.jar"],
    cmd = "$(location //third_party/ijar) --target_label //foo:foo --nostrip_jar $< $@",
    tools = ["//third_party/ijar"],
)

genrule(
    name = "jar_with_manifest_and_target_label_nostrip",
    srcs = ["jar-with-manifest-and-target-label.jar"],
    outs = ["jar-with-manifest-and-target-label-nostrip.jar"],
    cmd = "$(location //third_party/ijar) --target_label //foo:foo --nostrip_jar $< $@",
    tools = ["//third_party/ijar"],
)

genrule(
    name = "jar_without_manifest_nostrip",
    srcs = ["jar-without-manifest.jar"],
    outs = ["jar-without-manifest-nostrip.jar"],
    cmd = "$(location //third_party/ijar) --target_label //foo:foo --nostrip_jar $< $@",
    tools = ["//third_party/ijar"],
)

genrule(
    name = "jar_without_manifest_nostrip_idempotence",
    srcs = ["jar-without-manifest-nostrip.jar"],
    outs = ["jar-without-manifest-nostrip-idempotence.jar"],
    cmd = "$(location //third_party/ijar) --target_label //foo:foo --nostrip_jar $< $@",
    tools = ["//third_party/ijar"],
)

genrule(
    name = "empty_with_target_label",
    srcs = [":empty_zip.jar"],
    outs = ["empty_with_target_label.jar"],
    cmd = "$(location //third_party/ijar) $< $@ --target_label //empty",
    tools = ["//third_party/ijar"],
)

java_library(
    name = "typeannotations2",
    srcs = glob(["typeannotations2/**/*.java"]),
    tags = ["jdk8"],
)

java_library(
    name = "methodparameters",
    srcs = ["methodparameters/Test.java"],
    javacopts = ["-parameters"],
    tags = ["jdk8"],
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

java_library(
    name = "local_and_anonymous_lib",
    srcs = ["LocalAndAnonymous.java"],
)

genrule(
    name = "local_and_anonymous_interface",
    srcs = [":liblocal_and_anonymous_lib.jar"],
    outs = ["local_and_anonymous-interface.jar"],
    cmd = "$(location //third_party/ijar) $< $@",
    tools = ["//third_party/ijar"],
)

java_binary(
    name = "GenModuleInfo",
    testonly = 1,
    srcs = ["GenModuleInfo.java"],
    main_class = "GenModuleInfo",
    deps = ["//third_party:guava"],
)

genrule(
    name = "module_info",
    testonly = 1,
    outs = ["module_info.jar"],
    cmd = "$(location :GenModuleInfo) $@",
    tools = [":GenModuleInfo"],
)

genrule(
    name = "module_info_interface",
    testonly = 1,
    srcs = [":module_info.jar"],
    outs = ["module_info-interface.jar"],
    cmd = "$(location //third_party/ijar) $< $@",
    tools = ["//third_party/ijar"],
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
        "jar-with-manifest.jar",
        "jar-with-manifest-and-target-label.jar",
        "jar-without-manifest.jar",
        "package-info.java",
        ":empty_with_target_label",
        ":ijar_testlib",
        ":interface_ijar_testlib",
        ":interface_ijar_testlib_nostrip",
        ":interface_ijar_testlib_with_target_label",
        ":jar_with_manifest_and_target_label_nostrip",
        ":jar_with_manifest_nostrip",
        ":jar_without_manifest_nostrip",
        ":jar_without_manifest_nostrip_idempotence",
        ":liblocal_and_anonymous_lib.jar",
        ":local_and_anonymous-interface.jar",
        ":module_info-interface.jar",
        "//third_party/java/jdk/langtools:javac_jar",
    ],
    jvm_flags = [
        # Simulates how Bazel invokes javac
        "-Xbootclasspath/p:$(location //third_party/java/jdk/langtools:javac_jar)",
    ],
    tags = ["zip"],
    test_class = "IjarTests",
    deps = [
        "//src/java_tools/buildjar:BazelJavaCompiler",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
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
        ("empty_zip", 1),
    ]
]

java_binary(
    name = "zip_count",
    srcs = ["ZipCount.java"],
    jvm_flags = ["-Djdk.util.zip.inhibitZip64=false"],
    main_class = "test.ZipCount",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party/ijar:__pkg__"],
)
