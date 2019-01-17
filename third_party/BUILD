package(default_visibility = ["//visibility:public"])

load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//third_party/allocation_instrumenter:srcs",
        "//third_party/aws-sdk-auth-lite:srcs",
        "//third_party/aws-sig-v4-test-suite:srcs",
        "//third_party/css/bootstrap:srcs",
        "//third_party/css/font_awesome:srcs",
        "//third_party/ijar:srcs",
        "//third_party/iossim:srcs",
        "//third_party/grpc:srcs",
        "//third_party/jarjar:srcs",
        "//third_party/java/android_databinding:srcs",
        "//third_party/java/aosp_gradle_core:srcs",
        "//third_party/java/apkbuilder:srcs",
        "//third_party/java/dd_plist:srcs",
        "//third_party/java/j2objc:srcs",
        "//third_party/java/jacoco:srcs",
        "//third_party/java/javapoet:srcs",
        "//third_party/java/java_tools:srcs",
        "//third_party/java/jcommander:srcs",
        "//third_party/java/jdk/langtools:srcs",
        "//third_party/java/proguard:srcs",
        "//third_party/javascript/bootstrap:srcs",
        "//third_party/checker_framework_dataflow:srcs",
        "//third_party/checker_framework_javacutil:srcs",
        "//third_party/jformatstring:srcs",
        "//third_party/protobuf:srcs",
        "//third_party/py/abseil:srcs",
        "//third_party/py/concurrent:srcs",
        "//third_party/py/gflags:srcs",
        "//third_party/py/mock:srcs",
        "//third_party/py/six:srcs",
        "//third_party/zlib:srcs",
        "//third_party/nanopb:srcs",
        "//third_party/def_parser:srcs",
        "//third_party/pprof:srcs",
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
        "//third_party/checker_framework_dataflow:srcs",
        "//third_party/checker_framework_javacutil:srcs",
        "//third_party/jformatstring:srcs",
    ],
)

java_import(
    name = "aether",
    jars = [
        "aether/aether-api-1.0.0.v20140518.jar",
        "aether/aether-spi-1.0.0.v20140518.jar",
        "aether/aether-transport-http-1.0.0.v20140518.jar",
        "aether/aether-connector-basic-1.0.0.v20140518.jar",
        "aether/aether-transport-classpath-1.0.0.v20140518.jar",
        "aether/aether-transport-wagon-1.0.0.v20140518.jar",
        "aether/aether-impl-1.0.0.v20140518.jar",
        "aether/aether-transport-file-1.0.0.v20140518.jar",
        "aether/aether-util-1.0.0.v20140518.jar",
    ],
    deps = [
        ":apache_httpclient",
    ],
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
        "android_common/com.android.tools_ddmlib_25.0.0.jar",
        "android_common/com.android.tools_dvlib_25.0.0.jar",
        "android_common/com.android.tools_repository_25.0.0.jar",
        "android_common/com.android.tools_sdk-common_25.0.0.jar",
        "android_common/com.android.tools_sdklib_25.0.0.jar",
    ],
    runtime_deps = [
        ":asm",
        ":asm-analysis",
        ":asm-tree",
    ],
)

java_import(
    name = "apache_commons_codec",
    jars = ["apache_commons_codec/commons-codec-1.9.jar"],
)

java_import(
    name = "apache_commons_collections",
    jars = ["apache_commons_collections/commons-collections-3.2.2.jar"],
)

java_import(
    name = "apache_commons_lang",
    jars = ["apache_commons_lang/commons-lang-2.6.jar"],
)

java_import(
    name = "apache_commons_compress",
    jars = ["apache_commons_compress/apache-commons-compress-1.9.jar"],
)

java_import(
    name = "apache_commons_logging",
    jars = ["apache_commons_logging/commons-logging-1.1.1.jar"],
)

java_import(
    name = "apache_commons_pool2",
    jars = ["apache_commons_pool2/commons-pool2-2.3.jar"],
)

java_import(
    name = "apache_httpclient",
    jars = ["apache_httpclient/httpclient-4.5.3.jar"],
    deps = [
        ":apache_commons_codec",
        ":apache_commons_logging",
        ":apache_httpcore",
    ],
)

java_import(
    name = "apache_httpcore",
    jars = ["apache_httpcore/httpcore-4.4.6.jar"],
)

java_import(
    name = "apache_velocity",
    jars = ["apache_velocity/velocity-1.7.jar"],
    deps = [
        ":apache_commons_collections",
        ":apache_commons_lang",
    ],
)

java_import(
    name = "api_client",
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

java_import(
    name = "asm",
    jars = ["asm/asm-7.0.jar"],
    srcjar = "asm/asm-7.0-sources.jar",
)

java_import(
    name = "asm-analysis",
    jars = ["asm/asm-analysis-7.0.jar"],
    srcjar = "asm/asm-analysis-7.0-sources.jar",
    runtime_deps = [":asm-tree"],
)

java_import(
    name = "asm-commons",
    jars = ["asm/asm-commons-7.0.jar"],
    srcjar = "asm/asm-commons-7.0-sources.jar",
    runtime_deps = [":asm-tree"],
)

java_import(
    name = "asm-tree",
    jars = ["asm/asm-tree-7.0.jar"],
    srcjar = "asm/asm-tree-7.0-sources.jar",
    runtime_deps = [":asm"],
)

java_import(
    name = "asm-util",
    jars = ["asm/asm-util-7.0.jar"],
    srcjar = "asm/asm-util-7.0-sources.jar",
    runtime_deps = [":asm-tree"],
)

java_import(
    name = "auth",
    jars = [
        "auth/google-auth-library-oauth2-http-0.6.0.jar",
        "auth/google-auth-library-credentials-0.6.0.jar",
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

java_import(
    name = "auto_common",
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

java_import(
    name = "auto_service_lib",
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

java_import(
    name = "auto_value_value",
    jars = [
        "auto/auto-value-1.6.3rc1.jar",
        "auto/auto-value-annotations-1.6.3rc1.jar",
    ],
)

# For bootstrapping JavaBuilder
filegroup(
    name = "auto_value-jars",
    srcs = [
        "auto/auto-value-1.6.3rc1.jar",
        "auto/auto-value-annotations-1.6.3rc1.jar",
    ],
)

java_import(
    name = "compile_testing",
    jars = ["compile_testing/compile-testing-0.13.jar"],
)

java_import(
    name = "gson",
    jars = ["gson/gson-2.2.4.jar"],
)

java_import(
    name = "hungarian_algorithm",
    jars = ["hungarian_algorithm/software-and-algorithms-1.0.jar"],
    srcjar = "hungarian_algorithm/software-and-algorithms-1.0-src.jar",
)

java_import(
    name = "error_prone_annotations",
    jars = [
        "error_prone/error_prone_annotations-2.2.0.jar",
        "error_prone/error_prone_type_annotations-2.3.2-SNAPSHOT.jar",
    ],
)

java_import(
    name = "error_prone",
    jars = [
        "error_prone/error_prone_annotation-2.3.2-SNAPSHOT.jar",
        "error_prone/error_prone_check_api-2.3.2-SNAPSHOT.jar",
        "error_prone/error_prone_core-2.3.2-SNAPSHOT.jar",
    ],
    exports = [
        ":error_prone_annotations",
    ],
    deps = [
        ":auto_common",
        ":guava",
        ":hungarian_algorithm",
        ":jcip_annotations",
        ":jsr305",
        ":pcollections",
        "//third_party/checker_framework_dataflow",
        "//third_party/jformatstring",
    ],
)

java_import(
    name = "jackson2",
    jars = [
        "jackson2/jackson-core-2.8.6.jar",
    ],
)

java_import(
    name = "jcip_annotations",
    jars = [
        "jcip_annotations/jcip-annotations-1.0-1.jar",
    ],
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
        "error_prone/error_prone_annotations-2.2.0.jar",
        "guava/guava-25.1-jre.jar",
        "jcip_annotations/jcip-annotations-1.0-1.jar",
        "jsr305/jsr-305.jar",
    ],
)

java_import(
    name = "guava",
    jars = ["guava/guava-25.1-jre.jar"],
    exports = [
        ":error_prone_annotations",
        ":jcip_annotations",
        ":jsr305",
    ],
)

java_import(
    name = "flogger",
    jars = [
        "flogger/flogger-0.3.1.jar",
        "flogger/flogger-system-backend-0.3.1.jar",
        "flogger/google-extensions-0.3.1.jar",
    ],
)

filegroup(
    name = "flogger-jars",
    srcs = [
        "flogger/flogger-0.3.1.jar",
        "flogger/flogger-system-backend-0.3.1.jar",
        "flogger/google-extensions-0.3.1.jar",
    ],
)

java_import(
    name = "instrumentation",
    jars = [
        "instrumentation/instrumentation-api-0.4.3.jar",
    ],
)

java_import(
    name = "opencensus-api",
    jars = [
        "opencensus/opencensus-api-0.11.0.jar",
        "opencensus/opencensus-contrib-grpc-metrics-0.11.0.jar",
    ],
)

# For bootstrapping JavaBuilder
filegroup(
    name = "guava-jars",
    srcs = ["guava/guava-25.1-jre.jar"],
)

# javax.annotation.Generated is not included in the default root modules in 9,
# see: http://openjdk.java.net/jeps/320.
java_import(
    name = "javax_annotations",
    jars = ["javax_annotations/javax.annotation-api-1.3.2.jar"],
    neverlink = 1,  # @Generated is source-retention
    srcjar = "javax_annotations/javax.annotation-api-1.3.2-sources.jar",
)

java_import(
    name = "jimfs",
    jars = ["jimfs/jimfs-1.1.jar"],
)

java_import(
    name = "jsch",
    jars = ["jsch/jsch-0.1.54.jar"],
)

java_import(
    name = "jsr305",
    jars = ["jsr305/jsr-305.jar"],
)

# For bootstrapping JavaBuilder
filegroup(
    name = "jsr305-jars",
    srcs = ["jsr305/jsr-305.jar"],
)

java_import(
    name = "jsr330_inject",
    jars = ["jsr330_inject/javax.inject.jar"],
)

java_import(
    name = "maven",
    jars = glob(["maven/*.jar"]),
)

java_import(
    name = "maven_model",
    jars = [
        "maven_model/maven-aether-provider-3.2.3.jar",
        "maven_model/maven-model-3.2.3.jar",
        "maven_model/maven-model-builder-3.2.3.jar",
        "maven_model/maven-repository-metadata-3.2.3.jar",
    ],
)

java_import(
    name = "netty",
    jars = ["netty/netty-all-4.1.17.Final.jar"],
)

java_import(
    name = "netty_tcnative",
    jars = ["netty_tcnative/netty-tcnative-boringssl-static-2.0.7.Final.jar"],
)

java_import(
    name = "plexus_component_annotations",
    jars = ["plexus_component_annotations/plexus-component-annotations-1.6.jar"],
)

java_import(
    name = "plexus_interpolation",
    jars = ["plexus_interpolation/plexus-interpolation-1.22.jar"],
)

java_import(
    name = "plexus_utils",
    jars = ["plexus_utils/plexus-utils-3.0.21.jar"],
)

java_import(
    name = "slf4j",
    jars = [
        "slf4j/slf4j-api-1.7.7.jar",
        "slf4j/slf4j-jdk14-1.7.7.jar",
    ],
)

java_import(
    name = "tomcat_annotations_api",
    jars = ["tomcat_annotations_api/tomcat-annotations-api-8.0.5.jar"],
)

# For bootstrapping JavaBuilder
filegroup(
    name = "tomcat_annotations_api-jars",
    srcs = ["tomcat_annotations_api/tomcat-annotations-api-8.0.5.jar"],
)

# Testing

java_import(
    name = "diffutils",
    jars = ["diffutils/diffutils-1.3.0.jar"],
)

java_import(
    name = "guava-testlib",
    jars = ["guava/guava-testlib-25.1-jre.jar"],
)

java_import(
    name = "junit4",
    jars = [
        "hamcrest/hamcrest-core-1.3.jar",
        "junit/junit-4.12.jar",
    ],
)

java_import(
    name = "mockito",
    jars = ["mockito/mockito-all-1.10.19.jar"],
    deps = [
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
    jars = ["truth/truth-0.42.jar"],
    runtime_deps = [
        ":diffutils",
    ],
)

java_import(
    name = "truth8",
    jars = ["truth8/truth-java8-extension-0.42.jar"],
)

java_import(
    name = "xz",
    jars = ["xz/xz-1.5.jar"],
)

# To be used by the skylark example.
filegroup(
    name = "junit4-jars",
    srcs = [
        "hamcrest/hamcrest-core-1.3.jar",
        "junit/junit-4.12.jar",
    ],
)

pkg_tar(
    name = "java_tools_pkg",
    srcs = [
        "//third_party/java/java_tools:java_tools",
        "//third_party/java/jdk/langtools:javac_jar",
        "//third_party/java/jdk/langtools:jdk_compiler_jar",
        "//third_party/java/jdk/langtools:java_compiler_jar"
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
