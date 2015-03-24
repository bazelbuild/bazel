package(default_visibility = ["//visibility:public"])

load("/tools/build_rules/java_rules_skylark", "java_library")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "protoc",
    srcs = select({
        ":windows_mingw": ["protobuf/protoc.exe"],
        ":windows_msys64_mingw64": ["protobuf/protoc.exe"],
        ":windows_clang": ["protobuf/protoc.exe"],
        ":darwin": ["protobuf/protoc.darwin"],
        "//conditions:default": ["protobuf/protoc.amd64"],
    }),
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
    name = "apache_commons_compress",
    jars = ["apache_commons_compress/apache-commons-compress-1.9.jar"],
)

java_import(
    name = "apache_commons_logging",
    jars = ["apache_commons_logging/commons-logging-1.1.1.jar"],
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
    name = "gson",
    jars = ["gson/gson-2.2.4.jar"],
)

java_import(
    name = "guava",
    jars = ["guava/guava-18.0.jar"],
)

java_import(
    name = "joda_time",
    jars = ["joda_time/joda-time-2.3.jar"],
)

java_import(
    name = "jsr305",
    jars = ["jsr305/jsr-305.jar"],
)

java_import(
    name = "jsr330_inject",
    jars = ["jsr330_inject/javax.inject.jar"],
)

java_import(
    name = "maven_model",
    jars = [
        "maven_model/maven-aether-provider-3.2.3.jar",
        "maven_model/maven-model-3.2.3.jar",
        "maven_model/maven-model-builder-3.2.3.jar",
    ],
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
    name = "protobuf",
    jars = ["protobuf/protobuf-2.5.0.jar"],
)

# Testing

java_import(
    name = "guava-testlib",
    jars = ["guava/guava-testlib.jar"],
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
    name = "truth",
    jars = ["truth/truth-0.24+.jar"],
)

# For the java-skylark example.
java_library(
    name = "junit4-skylark",
    jars = [
        "hamcrest/hamcrest-core-1.3.jar",
        "junit/junit-4.11.jar",
    ],
)

java_import(
    name = "asm",
    jars = ["asm/asm-5.0.3.jar"],
)

config_setting(
    name = "windows_mingw",
    values = {"compiler": "windows_mingw"},
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
    values = {"cpu": "darwin"},
)
