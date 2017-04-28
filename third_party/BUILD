package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//third_party/css/bootstrap:srcs",
        "//third_party/css/font_awesome:srcs",
        "//third_party/ijar:srcs",
        "//third_party/iossim:srcs",
        "//third_party/grpc:srcs",
        "//third_party/java/android_databinding:srcs",
        "//third_party/java/aosp_gradle_core:srcs",
        "//third_party/java/apkbuilder:srcs",
        "//third_party/java/buck-ios-support:srcs",
        "//third_party/java/dd_plist:srcs",
        "//third_party/java/j2objc:srcs",
        "//third_party/java/jacoco:srcs",
        "//third_party/java/jarjar:srcs",
        "//third_party/java/javapoet:srcs",
        "//third_party/java/jopt-simple:srcs",
        "//third_party/java/jdk/langtools:srcs",
        "//third_party/java/jdk/javabuilder:srcs",
        "//third_party/javascript/bootstrap:srcs",
        "//third_party/checker_framework_dataflow:srcs",
        "//third_party/checker_framework_javacutil:srcs",
        "//third_party/jformatstring:srcs",
        "//third_party/protobuf:srcs",
        "//third_party/py/concurrent:srcs",
        "//third_party/py/gflags:srcs",
        "//third_party/py/mock:srcs",
        "//third_party/py/six:srcs",
        "//third_party/zlib:srcs",
        "//third_party/nanopb:srcs",
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

filegroup(
    name = "d3-js",
    srcs = glob(["javascript/d3/**/*.js"]),
)

filegroup(
    name = "jquery",
    srcs = glob(["javascript/jquery/**/*.jslib"]),
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
)

java_import(
    name = "android_common_25_0_0",
    jars = [
        "android_common/com.android_annotations_25.0.0.jar",
        "android_common/com.android.tools.build_builder_2.0.0.jar",
        "android_common/com.android.tools.build_builder-model_2.0.0.jar",
        "android_common/com.android.tools.build_builder-test-api_2.0.0.jar",
        "android_common/com.android.tools.build_manifest-merger_25.0.0.jar",
        "android_common/com.android.tools.external.lombok_lombok-ast_0.2.3.jar",
        "android_common/com.android.tools.layoutlib_layoutlib_25.0.0.jar",
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
    jars = ["apache_httpclient/httpclient-4.2.5.jar"],
)

java_import(
    name = "apache_httpcore",
    jars = ["apache_httpcore/httpcore-4.2.4.jar"],
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
        "api_client/google-http-client-1.22.0.jar",
        "api_client/google-http-client-jackson2-1.22.0.jar",
    ],
    runtime_deps = [
        ":jackson2",
    ],
)

java_import(
    name = "asm",
    jars = ["asm/asm-5.1.jar"],
)

java_import(
    name = "asm-analysis",
    jars = ["asm/asm-analysis-5.1.jar"],
    runtime_deps = [":asm-tree"],
)

java_import(
    name = "asm-commons",
    jars = ["asm/asm-commons-5.1.jar"],
    runtime_deps = [":asm-tree"],
)

java_import(
    name = "asm-tree",
    jars = ["asm/asm-tree-5.1.jar"],
    runtime_deps = [":asm"],
)

java_import(
    name = "asm-util",
    jars = ["asm/asm-util-5.1.jar"],
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
        ":auto_service",
        ":auto_value_value",
        ":guava",
        ":jsr305",
        ":tomcat_annotations_api",
    ],
)

java_import(
    name = "auto_common",
    jars = ["auto/auto-common-0.3.jar"],
)

java_import(
    name = "auto_service",
    jars = ["auto/auto-service-1.0-rc2.jar"],
)

java_plugin(
    name = "auto_service_plugin",
    processor_class = "com.google.auto.service.processor.AutoServiceProcessor",
    deps = [
        ":auto_common",
        ":auto_service",
        ":guava",
    ],
)

java_plugin(
    name = "auto_value_plugin",
    processor_class = "com.google.auto.value.processor.AutoValueProcessor",
    deps = [
        ":apache_commons_collections",
        ":apache_velocity",
        ":asm",
        ":auto_common",
        ":auto_service",
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
    jars = ["auto/auto-value-1.4.jar"],
)

# For bootstrapping JavaBuilder
filegroup(
    name = "auto_value-jars",
    srcs = ["auto/auto-value-1.4.jar"],
)

java_library(
    name = "dagger",
    exported_plugins = [":dagger_plugin"],
    exports = [
        ":dagger_api",
        ":jsr330_inject",
    ],
)

java_plugin(
    name = "dagger_plugin",
    processor_class = "dagger.internal.codegen.ComponentProcessor",
    deps = [
        ":dagger_api",
        ":dagger_compiler",
        ":dagger_producers",
        ":guava",
        ":jsr330_inject",
    ],
)

java_import(
    name = "dagger_api",
    jars = ["dagger/dagger-2.5.jar"],
)

java_import(
    name = "dagger_compiler",
    jars = ["dagger/dagger-compiler-2.5.jar"],
)

java_import(
    name = "dagger_producers",
    jars = ["dagger/dagger-producers-2.5.jar"],
)

java_import(
    name = "bytebuddy",
    jars = ["bytebuddy/byte-buddy-dep-0.7-rc6.jar"],
    deps = [
        ":asm",
        ":asm-commons",
        ":asm-util",
    ],
)

java_import(
    name = "gson",
    jars = ["gson/gson-2.2.4.jar"],
)

java_import(
    name = "hazelcast",
    jars = [
        "hazelcast/hazelcast-3.6.4.jar",
        "hazelcast/hazelcast-client-3.6.4.jar",
    ],
)

# TODO(cushon): delete
alias(
    name = "error_prone-2.0.20-SNAPSHOT",
    actual = ":error_prone",
)

java_import(
    name = "error_prone",
    jars = [
        "error_prone/error_prone_check_api-2.0.20-SNAPSHOT.jar",
        "error_prone/error_prone_core-2.0.20-SNAPSHOT.jar",
        "error_prone/error_prone_annotations-2.0.20-SNAPSHOT.jar",
        "error_prone/error_prone_annotation-2.0.20-SNAPSHOT.jar",
    ],
    exports = [
        ":auto_common",
        ":guava",
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
        "auto/auto-common-0.3.jar",
        "error_prone/error_prone_annotation-2.0.20-SNAPSHOT.jar",
        "error_prone/error_prone_annotations-2.0.20-SNAPSHOT.jar",
        "error_prone/error_prone_check_api-2.0.20-SNAPSHOT.jar",
        "error_prone/error_prone_core-2.0.20-SNAPSHOT.jar",
        "guava/guava-21.0.jar",
        "jcip_annotations/jcip-annotations-1.0-1.jar",
        "jsr305/jsr-305.jar",
        "pcollections/pcollections-2.1.2.jar",
        "//third_party/checker_framework_dataflow:bootstrap",
        "//third_party/jformatstring:bootstrap",
    ],
)

# TODO(cushon): delete
alias(
    name = "bootstrap_guava_and_error_prone-jars-2.0.20-SNAPSHOT",
    actual = "bootstrap_guava_and_error_prone-jars",
)

# As of Guava 20.0, Guava code uses Error Prone annotations. This isn't a
# problem when compiling with java 8, but is a problem when compiling Bazel
# with java 7 (the Error Prone jars need to be on the javac classpath). So,
# until we drop support for compiling Bazel with java 7, we unfortunately need
# to ensure that the Error Prone jars are always used together with the Guava
# jars.
java_import(
    name = "guava",
    jars = [
        "error_prone/error_prone_annotations-2.0.19.jar",
        "jcip_annotations/jcip-annotations-1.0-1.jar",
        "jsr305/jsr-305.jar",
    ] + select({
        "//tools/jdk:jdk7": ["guava/guava-21.0-20161101.jar"],
        "//conditions:default": ["guava/guava-21.0.jar"],
    }),
)

# For bootstrapping JavaBuilder
filegroup(
    name = "guava-jars",
    srcs = ["guava/guava-21.0.jar"],
)

java_import(
    name = "jgit",
    jars = ["jgit/org.eclipse.jgit-4.0.1.201506240215-r.jar"],
)

java_import(
    name = "jimfs",
    jars = ["jimfs/jimfs-1.1.jar"],
)

java_import(
    name = "joda_time",
    jars = ["joda_time/joda-time-2.3.jar"],
)

java_import(
    name = "jsch",
    jars = ["jsch/jsch-0.1.51.jar"],
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
    jars = ["netty/netty-all-4.1.0.CR6.jar"],
)

java_import(
    name = "netty_tcnative",
    jars = ["netty_tcnative/netty-tcnative-boringssl-static-1.1.33.Fork26.jar"],
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

# Testing

java_import(
    name = "guava-testlib",
    jars = select({
        "//tools/jdk:jdk7": ["guava/guava-testlib-21.0-20161101.jar"],
        "//conditions:default": ["guava/guava-testlib-21.0.jar"],
    }),
)

java_import(
    name = "junit4",
    jars = [
        "hamcrest/hamcrest-core-1.3.jar",
        "junit/junit-4.11.jar",
    ],
)

java_import(
    name = "mockito",
    jars = ["mockito/mockito-all-1.10.19.jar"],
)

java_import(
    name = "turbine",
    jars = ["turbine/turbine-0.1-20170427.jar"],
)

java_import(
    name = "truth",
    jars = ["truth/truth-0.32.jar"],
)

java_import(
    name = "xz",
    jars = ["xz/xz-1.5.jar"],
)

cc_library(
    name = "gtest",
    srcs = [
        "gtest/gtest-all.cc",
        "gtest/gtest_main.cc",
    ],
    hdrs = [
        "gtest/gtest.h",
    ],
    data = [
        "//tools:test_sharding_compliant",
    ],
    includes = ["."],
    # gtest detects pthread, and the linker fails without it.
    linkopts = ["-lpthread"],
)

# To be used by the skylark example.
filegroup(
    name = "junit4-jars",
    srcs = [
        "hamcrest/hamcrest-core-1.3.jar",
        "junit/junit-4.11.jar",
    ],
)

config_setting(
    name = "windows_mingw",
    values = {"compiler": "windows_mingw"},
)

config_setting(
    name = "windows_msys64",
    values = {"compiler": "windows_msys64"},
)

config_setting(
    name = "windows_msys64_mingw64",
    values = {"compiler": "windows_msys64_mingw64"},
)

config_setting(
    name = "windows_clang",
    values = {"compiler": "windows_clang"},
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
