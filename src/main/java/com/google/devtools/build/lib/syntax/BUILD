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

# The Starlark frontend (syntax, scanner, parser, resolver)
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
        "FileLocations.java",
        "FileOptions.java",
        "FlowStatement.java",
        "ForStatement.java",
        "Identifier.java",
        "IfStatement.java",
        "IndexExpression.java",
        "IntegerLiteral.java",
        "Lexer.java",
        "ListExpression.java",
        "LoadStatement.java",
        "Location.java",
        "Node.java",
        "NodePrinter.java",
        "NodeVisitor.java",
        "Parameter.java",
        "Parser.java",
        "ParserInput.java",
        "Resolver.java",
        "ReturnStatement.java",
        "SliceExpression.java",
        "StarlarkFile.java",
        "Statement.java",
        "StringLiteral.java",
        "SyntaxError.java",
        "TokenKind.java",
        "UnaryOperatorExpression.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/net/starlark/java/spelling",
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
        "BuiltinCallable.java",
        "CallUtils.java",
        "ClassObject.java",
        "CpuProfiler.java",
        "Debug.java",
        "Debugger.java",
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
        "Starlark.java",
        "StarlarkCallable.java",
        "StarlarkFunction.java",
        "StarlarkIndexable.java",
        "StarlarkIterable.java",
        "StarlarkList.java",
        "StarlarkQueryable.java",
        "StarlarkSemantics.java",
        "StarlarkThread.java",
        "StarlarkValue.java",
        "StringModule.java",
        "Tuple.java",
    ],
    data = [":libcpu_profiler.so"],  # will be loaded dynamically
    deps = [
        ":frontend",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/spelling",
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
        # There is no config_setting for Linux.
        # See https://github.com/bazelbuild/bazel/issues/11107
        "//src/conditions:linux_aarch64": ["cpu_profiler_posix.cc"],
        "//src/conditions:linux_arm": ["cpu_profiler_posix.cc"],
        "//src/conditions:linux_ppc": ["cpu_profiler_posix.cc"],
        "//src/conditions:linux_s390x": ["cpu_profiler_posix.cc"],
        "//src/conditions:linux_x86_64": ["cpu_profiler_posix.cc"],
        "//conditions:default": ["cpu_profiler_unimpl.cc"],
    }),
    linkshared = 1,
    deps = ["@bazel_tools//tools/jdk:jni"],
)
