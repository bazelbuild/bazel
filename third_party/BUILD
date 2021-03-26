load("@rules_java//java:defs.bzl", "java_import", "java_library", "java_plugin")
load("//tools/distributions:distribution_rules.bzl", "distrib_jar_filegroup", "distrib_java_import")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//third_party/allocation_instrumenter:srcs",
        "//third_party/animal_sniffer:srcs",
        "//third_party/antlr:srcs",
        "//third_party/aws-sdk-auth-lite:srcs",
        "//third_party/checker_framework_dataflow:srcs",
        "//third_party/checker_framework_javacutil:srcs",
        "//third_party/css/bootstrap:srcs",
        "//third_party/css/font_awesome:srcs",
        "//third_party/def_parser:srcs",
        "//third_party/grpc:srcs",
        "//third_party/ijar:srcs",
        "//third_party/jarjar:srcs",
        "//third_party/java/android_databinding:srcs",
        "//third_party/java/aosp_gradle_core:srcs",
        "//third_party/java/j2objc-annotations:srcs",
        "//third_party/java/j2objc:srcs",
        "//third_party/java/jacoco:srcs",
        "//third_party/java/javapoet:srcs",
        "//third_party/java/jcommander:srcs",
        "//third_party/java/jdk/langtools:srcs",
        "//third_party/java/proguard:srcs",
        "//third_party/javascript/bootstrap:srcs",
        "//third_party/jaxb:srcs",
        "//third_party/jetbrains_annotations:srcs",
        "//third_party/jetifier:srcs",
        "//third_party/jformatstring:srcs",
        "//third_party/juniversalchardet:srcs",
        "//third_party/kotlin_stdlib:srcs",
        "//third_party/pprof:srcs",
        "//third_party/protobuf:srcs",
        "//third_party/py/abseil:srcs",
        "//third_party/py/concurrent:srcs",
        "//third_party/py/dataclasses:srcs",
        "//third_party/py/frozendict:srcs",
        "//third_party/py/mock:srcs",
        "//third_party/py/six:srcs",
        "//third_party/rules_python:srcs",
        "//third_party/zlib:srcs",
        "@googleapis//:srcs",
        "@remoteapis//:srcs",
    ],
)

# Filegroup to ship the sources to the Bazel embededded tools
# This filegroup should contains all GPL with classpath exception
# and LGPL code that we use in Bazel.
filegroup(
    name = "gpl-srcs",
    srcs = [
        "//third_party/jformatstring:srcs",
    ],
)

# This target only contains the jars that are used for building / running Bazel.
# The target below is for the Android tools that are not shipped with Bazel.
distrib_java_import(
    name = "android_common_25_0_0_lite",
    enable_distributions = ["debian"],
    jars = [
        "android_common/com.android.tools.layoutlib_layoutlib_26.1.2-stripped.jar",
        "android_common/com.android.tools_sdk-common_25.0.0-patched-stripped.jar",
        "android_common/com.android.tools_repository_25.0.0.jar",
    ],
    deps = [
        "//third_party/jaxb",
    ],
)

java_import(
    name = "android-annotations",
    jars = ["android_common/com.android_annotations_25.0.0.jar"],
)

java_import(
    name = "android_common_25_0_0",
    jars = [
        "android_common/com.android_annotations_25.0.0.jar",
        "android_common/com.android.tools.build_builder_2.0.0.jar",
        "android_common/com.android.tools.build_builder-model_2.0.0.jar",
        "android_common/com.android.tools.build_builder-test-api_2.0.0.jar",
        "android_common/com.android.tools.build_manifest-merger_25.0.0-patched.jar",
        "android_common/com.android.tools.external.lombok_lombok-ast_0.2.3.jar",
        # layoutlib has been upgraded to 26.1.2 *just* for font resource support.
        # We are unable to upgrade the other libraries yet due to internal
        # constraints with Blaze. See
        # https://github.com/bazelbuild/bazel/issues/4381
        # TODO(bazel-team): Upgrade entire android_common suite to 26+ after
        # resolving internal constraint.
        "android_common/com.android.tools.layoutlib_layoutlib_26.1.2.jar",
        "android_common/com.android.tools.lint_lint-api_25.0.0.jar",
        "android_common/com.android.tools.lint_lint-checks_25.0.0.jar",
        "android_common/com.android.tools_common_25.0.0.jar",
        "android_common/com.android.tools_dvlib_25.0.0.jar",
        "android_common/com.android.tools_repository_25.0.0.jar",
        "android_common/com.android.tools_sdk-common_25.0.0-patched.jar",
        "android_common/com.android.tools_sdklib_25.0.0.jar",
    ],
    runtime_deps = [
        ":asm",
        ":asm-analysis",
        ":asm-tree",
    ],
    deps = [
        "//third_party/jaxb",
    ],
)

java_import(
    name = "apache_commons_codec",
    jars = ["apache_commons_codec/commons-codec-1.10.jar"],
)

distrib_java_import(
    name = "apache_commons_collections",
    enable_distributions = ["debian"],
    jars = ["apache_commons_collections/commons-collections-3.2.2.jar"],
)

java_import(
    name = "apache_commons_io",
    jars = ["apache_commons_io/commons-io-2.4.jar"],
)

distrib_java_import(
    name = "apache_commons_lang",
    enable_distributions = ["debian"],
    jars = ["apache_commons_lang/commons-lang-2.6.jar"],
)

distrib_java_import(
    name = "apache_commons_compress",
    enable_distributions = ["debian"],
    jars = ["apache_commons_compress/apache-commons-compress-1.19.jar"],
)

distrib_java_import(
    name = "apache_commons_logging",
    enable_distributions = ["debian"],
    jars = ["apache_commons_logging/commons-logging-1.1.1.jar"],
)

distrib_java_import(
    name = "apache_commons_pool2",
    enable_distributions = ["debian"],
    jars = ["apache_commons_pool2/commons-pool2-2.8.0.jar"],
)

distrib_java_import(
    name = "apache_velocity",
    enable_distributions = ["debian"],
    jars = ["apache_velocity/velocity-1.7.jar"],
    deps = [
        ":apache_commons_collections",
        ":apache_commons_lang",
    ],
)

distrib_java_import(
    name = "api_client",
    enable_distributions = ["debian"],
    jars = [
        "api_client/google-api-client-1.22.0.jar",
        "api_client/google-api-client-jackson2-1.22.0.jar",
        "api_client/google-http-client-1.22.0-SNAPSHOT.jar",
        "api_client/google-http-client-jackson2-1.22.0.jar",
    ],
    runtime_deps = [
        ":jackson2",
    ],
)

distrib_java_import(
    name = "asm",
    enable_distributions = ["debian"],
    jars = ["asm/asm-8.0.jar"],
    srcjar = "asm/asm-8.0-sources.jar",
)

java_import(
    name = "asm-analysis",
    jars = ["asm/asm-analysis-8.0.jar"],
    srcjar = "asm/asm-analysis-8.0-sources.jar",
    runtime_deps = [":asm-tree"],
)

java_import(
    name = "asm-commons",
    jars = ["asm/asm-commons-8.0.jar"],
    srcjar = "asm/asm-commons-8.0-sources.jar",
    runtime_deps = [":asm-tree"],
)

java_import(
    name = "asm-tree",
    jars = ["asm/asm-tree-8.0.jar"],
    srcjar = "asm/asm-tree-8.0-sources.jar",
    runtime_deps = [":asm"],
)

java_import(
    name = "asm-util",
    jars = ["asm/asm-util-8.0.jar"],
    srcjar = "asm/asm-util-8.0-sources.jar",
    runtime_deps = [":asm-tree"],
)

distrib_java_import(
    name = "auth",
    enable_distributions = ["debian"],
    jars = [
        "auth/google-auth-library-oauth2-http-0.17.1.jar",
        "auth/google-auth-library-credentials-0.17.1.jar",
    ],
    runtime_deps = [
        ":api_client",
        ":guava",
        "//third_party/aws-sdk-auth-lite",
    ],
)

java_plugin(
    name = "auto_annotation_plugin",
    processor_class = "com.google.auto.value.processor.AutoAnnotationProcessor",
    deps = [
        ":apache_commons_collections",
        ":apache_velocity",
        ":asm",
        ":auto_common",
        ":auto_service_lib",
        ":auto_value_value",
        ":guava",
        ":jsr305",
        ":tomcat_annotations_api",
    ],
)

distrib_java_import(
    name = "auto_common",
    enable_distributions = ["debian"],
    jars = ["auto/auto-common-0.10.jar"],
)

java_library(
    name = "auto_service",
    exported_plugins = [
        ":auto_service_plugin",
    ],
    exports = [
        ":auto_service_lib",
    ],
)

java_plugin(
    name = "auto_service_plugin",
    processor_class = "com.google.auto.service.processor.AutoServiceProcessor",
    deps = [
        ":auto_common",
        ":auto_service_lib",
        ":guava",
    ],
)

distrib_java_import(
    name = "auto_service_lib",
    enable_distributions = ["debian"],
    jars = ["auto/auto-service-1.0-rc4.jar"],
)

java_plugin(
    name = "auto_value_plugin",
    processor_class = "com.google.auto.value.processor.AutoValueProcessor",
    deps = [
        ":apache_commons_collections",
        ":apache_velocity",
        ":asm",
        ":auto_common",
        ":auto_service_lib",
        ":auto_value_value",
        ":guava",
        ":tomcat_annotations_api",
    ],
)

java_library(
    name = "auto_value",
    exported_plugins = [
        ":auto_annotation_plugin",
        ":auto_value_plugin",
    ],
    exports = [
        ":auto_value_value",
        ":tomcat_annotations_api",
    ],
)

distrib_java_import(
    name = "auto_value_value",
    enable_distributions = ["debian"],
    jars = [
        "auto/auto-value-1.6.3rc1.jar",
        "auto/auto-value-annotations-1.6.3rc1.jar",
    ],
)

# For bootstrapping JavaBuilder
distrib_jar_filegroup(
    name = "auto_value-jars",
    srcs = [
        "auto/auto-value-1.6.3rc1.jar",
        "auto/auto-value-annotations-1.6.3rc1.jar",
    ],
    enable_distributions = ["debian"],
)

java_import(
    name = "byte_buddy",
    jars = [
        "bytebuddy/byte-buddy-1.9.7.jar",
        "bytebuddy/byte-buddy-agent-1.9.7.jar",
    ],
)

distrib_java_import(
    name = "checker_framework_annotations",
    enable_distributions = ["debian"],
    jars = ["checker_framework_annotations/checker-qual-3.2.0.jar"],
    srcjar = "checker_framework_annotations/checker-qual-3.2.0-sources.jar",
)

java_import(
    name = "compile_testing",
    jars = ["compile_testing/compile-testing-0.18.jar"],
)

distrib_java_import(
    name = "gson",
    enable_distributions = ["debian"],
    jars = ["gson/gson-2.8.0.jar"],
)

java_import(
    name = "caffeine",
    jars = ["caffeine/caffeine-2.8.0.jar"],
)

java_import(
    name = "hungarian_algorithm",
    jars = ["hungarian_algorithm/software-and-algorithms-1.0.jar"],
    srcjar = "hungarian_algorithm/software-and-algorithms-1.0-src.jar",
)

java_import(
    name = "threeten",
    jars = [
        "error_prone/threeten-extra-1.5.0.jar",
    ],
)

distrib_java_import(
    name = "error_prone_annotations",
    enable_distributions = ["debian"],
    jars = [
        "error_prone/error_prone_annotations-2.5.1.jar",
        "error_prone/error_prone_type_annotations-2.5.1.jar",
    ],
)

distrib_jar_filegroup(
    name = "error_prone_annotations-jar",
    srcs = [
        "error_prone/error_prone_annotations-2.5.1.jar",
        "error_prone/threeten-extra-1.5.0.jar",
    ],
    enable_distributions = ["debian"],
)

java_import(
    name = "error_prone",
    jars = [
        "error_prone/error_prone_annotation-2.5.1.jar",
        "error_prone/error_prone_check_api-2.5.1.jar",
        "error_prone/error_prone_core-2.5.1.jar",
    ],
    exports = [
        ":error_prone_annotations",
    ],
    deps = [
        ":auto_common",
        ":caffeine",
        ":guava",
        ":hungarian_algorithm",
        ":jcip_annotations",
        ":jsr305",
        ":pcollections",
        ":threeten",
        "//third_party/checker_framework_dataflow",
        "//third_party/jformatstring",
    ],
)

distrib_java_import(
    name = "jackson2",
    enable_distributions = ["debian"],
    jars = [
        "jackson2/jackson-core-2.8.6.jar",
    ],
)

distrib_java_import(
    name = "jcip_annotations",
    enable_distributions = ["debian"],
    jars = [
        "jcip_annotations/jcip-annotations-1.0-1.jar",
    ],
)

# For bootstrapping JavaBuilder
distrib_jar_filegroup(
    name = "jcip_annotations-jars",
    srcs = [
        "jcip_annotations/jcip-annotations-1.0-1.jar",
    ],
    enable_distributions = ["debian"],
)

java_import(
    name = "pcollections",
    jars = [
        "pcollections/pcollections-2.1.2.jar",
    ],
)

# For bootstrapping JavaBuilder
filegroup(
    name = "bootstrap_guava_and_error_prone-jars",
    srcs = [
        ":error_prone_annotations-jar",
        ":guava-jars",
        ":jcip_annotations-jars",
        ":jsr305-jars",
    ],
)

distrib_java_import(
    name = "guava",
    enable_distributions = ["debian"],
    jars = [
        "guava/failureaccess-1.0.1.jar",
        "guava/guava-29.0-jre.jar",
    ],
    exports = [
        ":error_prone_annotations",
        ":jcip_annotations",
        ":jsr305",
    ],
)

distrib_java_import(
    name = "flogger",
    enable_distributions = ["debian"],
    jars = [
        "flogger/flogger-0.5.1.jar",
        "flogger/flogger-system-backend-0.5.1.jar",
        "flogger/google-extensions-0.5.1.jar",
    ],
)

distrib_jar_filegroup(
    name = "flogger-jars",
    srcs = [
        "flogger/flogger-0.5.1.jar",
        "flogger/flogger-system-backend-0.5.1.jar",
        "flogger/google-extensions-0.5.1.jar",
    ],
    enable_distributions = ["debian"],
)

distrib_java_import(
    name = "opencensus-api",
    enable_distributions = ["debian"],
    jars = [
        "opencensus/opencensus-api-0.24.0.jar",
        "opencensus/opencensus-contrib-grpc-metrics-0.24.0.jar",
    ],
)

distrib_java_import(
    name = "perfmark-api",
    enable_distributions = ["debian"],
    jars = [
        "perfmark/perfmark-api-0.19.0.jar",
    ],
)

# For bootstrapping JavaBuilder
distrib_jar_filegroup(
    name = "guava-jars",
    srcs = ["guava/guava-29.0-jre.jar"],
    enable_distributions = ["debian"],
)

java_import(
    name = "javax_activation",
    jars = ["javax_activation/javax.activation-api-1.2.0.jar"],
    srcjar = "javax_activation/javax.activation-api-1.2.0-sources.jar",
)

# javax.annotation.Generated is not included in the default root modules in 9,
# see: http://openjdk.java.net/jeps/320.
distrib_java_import(
    name = "javax_annotations",
    enable_distributions = ["debian"],
    jars = ["javax_annotations/javax.annotation-api-1.3.2.jar"],
    neverlink = 1,  # @Generated is source-retention
    srcjar = "javax_annotations/javax.annotation-api-1.3.2-sources.jar",
)

java_import(
    name = "reactive_streams",
    jars = ["reactive_streams/reactive-streams-1.0.3.jar"],
)

java_import(
    name = "rxjava3",
    jars = ["rxjava3/rxjava-3.0.9.jar"],
    exports = [":reactive_streams"],
    deps = [":reactive_streams"],
)

java_import(
    name = "jimfs",
    jars = ["jimfs/jimfs-1.1.jar"],
)

distrib_java_import(
    name = "jsr305",
    enable_distributions = ["debian"],
    jars = ["jsr305/jsr-305.jar"],
)

# For bootstrapping JavaBuilder
distrib_jar_filegroup(
    name = "jsr305-jars",
    srcs = ["jsr305/jsr-305.jar"],
    enable_distributions = ["debian"],
)

java_import(
    name = "jsr330_inject",
    jars = ["jsr330_inject/javax.inject.jar"],
)

UNNECESSARY_DYNAMIC_LIBRARIES = select({
    "//src/conditions:windows": "*.so *.jnilib",
    "//src/conditions:darwin": "*.so *.dll",
    "//src/conditions:linux_x86_64": "*.jnilib *.dll",
    # The .so file is an x86 one, so we can just remove it if the CPU is not x86
    "//src/conditions:arm": "*.so *.jnilib *.dll",
    "//src/conditions:linux_aarch64": "*.so *.jnilib *.dll",
    "//src/conditions:linux_ppc": "*.so *.jnilib *.dll",
    "//src/conditions:linux_s390x": "*.so *.jnilib *.dll",
    "//src/conditions:freebsd": "*.so *.jnilib *.dll",
    "//src/conditions:openbsd": "*.so *.jnilib *.dll",
    # Default is to play it safe -- better have a big binary than a slow binary
    #   The empty string means nothing is to be removed from the library;
    #   the rule command tests for the empty string explictly to avoid
    #   zip erroring when it finds nothing to remove.
    "//conditions:default": "",
})

# Remove native libraries that are for a platform different from the one we are
# building Bazel for.
genrule(
    name = "filter_netty_dynamic_libs",
    srcs = ["netty_tcnative/netty-tcnative-boringssl-static-2.0.24.Final.jar"],
    outs = ["netty_tcnative/netty-tcnative-filtered.jar"],
    cmd = "cp $< $@ && " +
          # Make sure we can write the output file, even if the input isn't writable.
          "chmod +w $@ && " +
          # End successfully if there is nothing to be deleted from the archive
          "if [ -n '" + UNNECESSARY_DYNAMIC_LIBRARIES + "' ]; then " +
          "zip -qd $@ " + UNNECESSARY_DYNAMIC_LIBRARIES + "; fi",
)

distrib_java_import(
    name = "netty",
    enable_distributions = ["debian"],
    jars = ["netty/netty-all-4.1.48.Final.jar"],
)

distrib_java_import(
    name = "netty_tcnative",
    enable_distributions = ["debian"],
    jars = ["netty_tcnative/netty-tcnative-filtered.jar"],
)

distrib_java_import(
    name = "tomcat_annotations_api",
    enable_distributions = ["debian"],
    jars = ["tomcat_annotations_api/tomcat-annotations-api-8.0.5.jar"],
)

# For bootstrapping JavaBuilder
distrib_jar_filegroup(
    name = "tomcat_annotations_api-jars",
    srcs = ["tomcat_annotations_api/tomcat-annotations-api-8.0.5.jar"],
    enable_distributions = ["debian"],
)

distrib_java_import(
    name = "java-diff-utils",
    enable_distributions = ["debian"],
    jars = ["java-diff-utils/java-diff-utils-4.0.jar"],
)

# Testing

# TODO: Remove this target after migrating to java-diff-utils
java_import(
    name = "diffutils",
    jars = ["diffutils/diffutils-1.3.0.jar"],
)

java_import(
    name = "guava-testlib",
    jars = ["guava/guava-testlib-29.0-jre.jar"],
)

distrib_java_import(
    name = "junit4",
    enable_distributions = ["debian"],
    jars = [
        "hamcrest/hamcrest-core-1.3.jar",
        "junit/junit-4.13.jar",
    ],
)

java_import(
    name = "mockito",
    jars = ["mockito/mockito-core-2.25.1.jar"],
    deps = [
        ":byte_buddy",
        ":objenesis",
    ],
)

java_import(
    name = "objenesis",
    jars = ["objenesis/objenesis-1_3.jar"],
)

filegroup(
    name = "turbine_direct",
    srcs = ["turbine/turbine_direct.jar"],
)

java_import(
    name = "turbine",
    jars = ["turbine/turbine_direct.jar"],
)

java_import(
    name = "truth",
    jars = ["truth/truth-1.0.1.jar"],
    exports = [
        ":truth8",
        ":truth_proto",
    ],
    deps = [
        ":diffutils",
    ],
)

java_import(
    name = "truth8",
    jars = ["truth8/truth-java8-extension-1.0.1.jar"],
)

java_import(
    name = "truth_proto",
    jars = [
        "truth_proto/truth-liteproto-extension-1.0.1.jar",
        "truth_proto/truth-proto-extension-1.0.1.jar",
    ],
)

distrib_java_import(
    name = "xz",
    enable_distributions = ["debian"],
    jars = ["xz/xz-1.5.jar"],
)

# To be used by the skylark example.
filegroup(
    name = "junit4-jars",
    srcs = [
        "hamcrest/hamcrest-core-1.3.jar",
        "junit/junit-4.13.jar",
    ],
)

load(":compiler_config_setting.bzl", "create_compiler_config_setting")

create_compiler_config_setting(
    name = "windows_mingw",
    value = "windows_mingw",
)

create_compiler_config_setting(
    name = "windows_msys64",
    value = "windows_msys64",
)

create_compiler_config_setting(
    name = "windows_msys64_mingw64",
    value = "windows_msys64_mingw64",
)

create_compiler_config_setting(
    name = "windows_clang",
    value = "windows_clang",
)

config_setting(
    name = "darwin",
    values = {"host_cpu": "darwin"},
)

config_setting(
    name = "k8",
    values = {"host_cpu": "k8"},
)

config_setting(
    name = "piii",
    values = {"host_cpu": "piii"},
)

config_setting(
    name = "arm",
    values = {"host_cpu": "arm"},
)

config_setting(
    name = "aarch64",
    values = {"host_cpu": "aarch64"},
)

config_setting(
    name = "freebsd",
    values = {"host_cpu": "freebsd"},
)

config_setting(
    name = "openbsd",
    values = {"host_cpu": "openbsd"},
)

config_setting(
    name = "s390x",
    values = {"host_cpu": "s390x"},
)

config_setting(
    name = "ppc",
    values = {"host_cpu": "ppc"},
)

test_suite(
    name = "all_windows_tests",
    tests = [
        "//third_party/def_parser:windows_tests",
    ],
    visibility = ["//src:__pkg__"],
)
