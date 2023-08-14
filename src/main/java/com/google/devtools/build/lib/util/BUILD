load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/util/io:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "logging",
    srcs = ["LoggingUtil.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "var_int",
    srcs = [
        "VarInt.java",
    ],
)

java_library(
    name = "os",
    srcs = ["OS.java"],
)

java_library(
    name = "classpath",
    srcs = ["Classpath.java"],
    deps = ["//third_party:guava"],
)

java_library(
    name = "process",
    srcs = ["ProcessUtils.java"],
    deps = [
        ":os",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/windows:processes",
    ],
)

java_library(
    name = "single_line_formatter",
    srcs = ["SingleLineFormatter.java"],
    deps = [
        "//third_party:guava",
    ],
)

java_library(
    name = "simple_log_handler",
    srcs = ["SimpleLogHandler.java"],
    deps = [
        ":os",
        ":single_line_formatter",
        ":util",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "debug-logger-configurator",
    srcs = ["DebugLoggerConfigurator.java"],
    deps = [":util"],
)

java_library(
    name = "describable_execution_unit",
    srcs = ["DescribableExecutionUnit.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "command",
    srcs = [
        "CommandBuilder.java",
        "CommandDescriptionForm.java",
        "CommandFailureUtils.java",
        "CommandUtils.java",
    ],
    deps = [
        ":describable_execution_unit",
        ":os",
        ":shell_escaper",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "maybe_complete_set",
    srcs = ["MaybeCompleteSet.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "resource_converter",
    srcs = [
        "ResourceConverter.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "cpu_resource_converter",
    srcs = [
        "CpuResourceConverter.java",
    ],
    deps = [
        ":resource_converter",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//third_party:guava",
    ],
)

java_library(
    name = "ram_resource_converter",
    srcs = [
        "RamResourceConverter.java",
    ],
    deps = [
        ":resource_converter",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//third_party:guava",
    ],
)

java_library(
    name = "pair",
    srcs = [
        "Pair.java",
    ],
    deps = [
        "//third_party:jsr305",
    ],
)

java_library(
    name = "util",
    srcs = [
        "AbstractIndexer.java",
        "AnsiStrippingOutputStream.java",
        "CPU.java",
        "CanonicalStringIndexer.java",
        "ClassName.java",
        "DependencySet.java",
        "Either.java",
        "FileHandlerQuerier.java",
        "Fingerprint.java",
        "JavaSleeper.java",
        "LogHandlerQuerier.java",
        "LoggingUtil.java",
        "LongArrayList.java",
        "NetUtil.java",
        "OnDemandString.java",
        "OptionsUtils.java",
        "OrderedSetMultimap.java",
        "OsUtils.java",
        "PathFragmentFilter.java",
        "PersistentMap.java",
        "RegexFilter.java",
        "ResourceFileLoader.java",
        "Sleeper.java",
        "StreamWriter.java",
        "StringIndexer.java",
        "ThreadUtils.java",
        "TimeUtilities.java",
        "UserUtils.java",
    ],
    exports = [
        # vfs depends on the profiler and creates a cycle since we use Pair in profiler
        ":pair",
    ],
    deps = [
        ":os",
        ":shell_escaper",
        ":string",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:error_prone_annotations",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "filetype",
    srcs = [
        "FileType.java",
        "FileTypeSet.java",
    ],
    deps = [
        ":string",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/vfs:ospathpolicy",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "TestType",
    srcs = ["TestType.java"],
)

java_library(
    name = "resource_usage",
    srcs = [
        "ResourceUsage.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/unix:procmeminfo_parser",
        "//third_party:guava",
    ],
)

java_library(
    name = "shell_escaper",
    srcs = [
        "GccParamFileEscaper.java",
        "ShellEscaper.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "string",
    srcs = [
        "StringCanonicalizer.java",
        "StringUtil.java",
        "StringUtilities.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
    ],
)

java_library(
    name = "abrupt_exit_exception",
    srcs = ["AbruptExitException.java"],
    deps = [
        ":detailed_exit_code",
        ":exit_code",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "exit_code",
    srcs = [
        "ExitCode.java",
    ],
    deps = [
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "custom_exit_code_publisher",
    srcs = ["CustomExitCodePublisher.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "custom_failure_detail_publisher",
    srcs = ["CustomFailureDetailPublisher.java"],
    deps = [
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "detailed_exit_code",
    srcs = ["DetailedExitCode.java"],
    deps = [
        ":exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "crash_failure_details",
    srcs = ["CrashFailureDetails.java"],
    deps = [
        ":detailed_exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:flogger",
        "//third_party:guava",
    ],
)

java_library(
    name = "hash_codes",
    srcs = ["HashCodes.java"],
    deps = ["//third_party:jsr305"],
)

java_library(
    name = "interrupted_failure_details",
    srcs = ["InterruptedFailureDetails.java"],
    deps = [
        ":abrupt_exit_exception",
        ":detailed_exit_code",
        "//src/main/protobuf:failure_details_java_proto",
    ],
)
