# Description: Bazel's Starlark interpreter
#
# This BUILD file defines two java_library rules that
# define a single Java package in two parts,
# the Starlark parser and evaluator.
# Eventually the Java package will be split in two,
# but splitting only the build rules helps enforce
# that only evaluator -> frontend dependencies are permitted,
# without the churn to clients of renaming classes one at a time
# as they move between two different Java packages.
#
# Until the split, clients should depend only on
# //third_party/bazel/src/main/java/com/google/devtools/build/lib:syntax,
# which is a facade for the complete package.

load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

# The Starlark frontend (syntax, scanner, parser, validator)
java_library(
    name = "frontend",
    srcs = [
        "Argument.java",
        "AssignmentStatement.java",
        "BinaryOperatorExpression.java",
        "CallExpression.java",
        "Comment.java",
        "Comprehension.java",
        "ConditionalExpression.java",
        "DefStatement.java",
        "DictExpression.java",
        "DotExpression.java",
        "Expression.java",
        "ExpressionStatement.java",
        "FileOptions.java",
        "FlowStatement.java",
        "ForStatement.java",
        "FunctionSignature.java",
        "Identifier.java",
        "IfStatement.java",
        "IndexExpression.java",
        "IntegerLiteral.java",
        "Lexer.java",
        "LineNumberTable.java",
        "ListExpression.java",
        "LoadStatement.java",
        "Location.java",
        "Node.java",
        "NodePrinter.java",
        "NodeVisitor.java",
        "Parameter.java",
        "Parser.java",
        "ParserInput.java",
        "ReturnStatement.java",
        "SliceExpression.java",
        "StarlarkFile.java",
        "Statement.java",
        "StringLiteral.java",
        "SyntaxError.java",
        "Token.java",
        "TokenKind.java",
        "UnaryOperatorExpression.java",
        "ValidationEnvironment.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/starlark/spelling",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

# The Starlark evaluator
java_library(
    name = "evaluator",
    srcs = [
        "BaseFunction.java",
        "BuiltinCallable.java",
        "CallUtils.java",
        "ClassObject.java",
        "CpuProfiler.java",
        "Debug.java",
        "Debugger.java",
        "Depset.java",
        "Dict.java",
        "Eval.java",
        "EvalException.java",
        "EvalExceptionWithStackTrace.java",
        "EvalUtils.java",
        "FlagGuardedValue.java",
        "FormatParser.java",
        "HasBinary.java",
        "JNI.java",
        "MethodDescriptor.java",
        "MethodLibrary.java",
        "Module.java",
        "Mutability.java",
        "NoneType.java",
        "ParamDescriptor.java",
        "Printer.java",
        "RangeList.java",
        "Sequence.java",
        "SkylarkClassObject.java",
        "SkylarkIndexable.java",
        "SkylarkQueryable.java",
        "SkylarkType.java",
        "Starlark.java",
        "StarlarkCallable.java",
        "StarlarkFunction.java",
        "StarlarkIterable.java",
        "StarlarkList.java",
        "StarlarkSemantics.java",
        "StarlarkThread.java",
        "StarlarkValue.java",
        "StringModule.java",
        "Tuple.java",
    ],
    data = [":libcpu_profiler.so"],  # will be loaded dynamically
    deps = [
        ":frontend",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/starlark/spelling",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

# The C++ portion of the Starlark CPU profiler.
cc_binary(
    name = "libcpu_profiler.so",
    srcs = select({
        "//src/conditions:darwin": ["cpu_profiler_posix.cc"],
        "//src/conditions:linux_x86_64": ["cpu_profiler_posix.cc"],
        "//conditions:default": ["cpu_profiler_unimpl.cc"],
    }),
    linkshared = 1,
    deps = [":jni"],  # TODO(adonovan): use @bazel_tools//tools/jdk:jni when released
)

# This has been copied to @bazel_tools//tools/jdk:jni and will appear in the next release.
cc_library(
    name = "jni",
    hdrs = ["@bazel_tools//tools/jdk:jni_header"] + select({
        "//src/conditions:linux_x86_64": ["@bazel_tools//tools/jdk:jni_md_header-linux"],
        "//src/conditions:linux_aarch64": ["@bazel_tools//tools/jdk:jni_md_header-linux"],
        "//src/conditions:linux_ppc": ["@bazel_tools//tools/jdk:jni_md_header-linux"],
        "//src/conditions:darwin": ["@bazel_tools//tools/jdk:jni_md_header-darwin"],
        "//src/conditions:freebsd": ["@bazel_tools//tools/jdk:jni_md_header-freebsd"],
        "//src/conditions:openbsd": ["@bazel_tools//tools/jdk:jni_md_header-openbsd"],
        "//src/conditions:windows": ["@bazel_tools//tools/jdk:jni_md_header-windows"],
        "//conditions:default": [],
    }),
    includes = ["../../../../../../../../../external/bazel_tools/tools/jdk/include"] + select({
        "//src/conditions:linux_x86_64": ["../../../../../../../../../external/bazel_tools/tools/jdk/include/linux"],
        "//src/conditions:linux_aarch64": ["../../../../../../../../../external/bazel_tools/tools/jdk/include/linux"],
        "//src/conditions:linux_ppc": ["../../../../../../../../../external/bazel_tools/tools/jdk/include/linux"],
        "//src/conditions:darwin": ["../../../../../../../../../external/bazel_tools/tools/jdk/include/darwin"],
        "//src/conditions:freebsd": ["../../../../../../../../../external/bazel_tools/tools/jdk/include/freebsd"],
        "//src/conditions:openbsd": ["../../../../../../../../../external/bazel_tools/tools/jdk/include/openbsd"],
        "//src/conditions:windows": ["../../../../../../../../../external/bazel_tools/tools/jdk/include/win32"],
        "//conditions:default": [],
    }),
)
