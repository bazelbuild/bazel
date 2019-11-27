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
        "Comment.java",
        "Comprehension.java",
        "ConditionalExpression.java",
        "DefStatement.java",
        "DictExpression.java",
        "DotExpression.java",
        "Expression.java",
        "ExpressionStatement.java",
        "FlowStatement.java",
        "ForStatement.java",
        "FuncallExpression.java",
        "FunctionSignature.java",
        "Identifier.java",
        "IfStatement.java",
        "IndexExpression.java",
        "IntegerLiteral.java",
        "Lexer.java",
        "LineNumberTable.java",
        "ListExpression.java",
        "LoadStatement.java",
        "Node.java",
        "NodePrinter.java",
        "NodeVisitor.java",
        "Parameter.java",
        "Parser.java",
        "ParserInput.java",
        "ReturnStatement.java",
        "SliceExpression.java",
        "StarlarkFile.java",
        "StarlarkSemantics.java",
        "Statement.java",
        "StringLiteral.java",
        "SyntaxError.java",
        "Token.java",
        "TokenKind.java",
        "UnaryOperatorExpression.java",
        "ValidationEnvironment.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:string_util",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
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
        "BuiltinFunction.java",
        "CallUtils.java",
        "Callstack.java",
        "ClassObject.java",
        "Concatable.java",
        "DebugFrame.java",
        "Debugger.java",
        "Depset.java",
        "Dict.java",
        "Eval.java",
        "EvalException.java",
        "EvalExceptionWithStackTrace.java",
        "EvalUtils.java",
        "FlagGuardedValue.java",
        "FormatParser.java",
        "MethodDescriptor.java",
        "MethodLibrary.java",
        "Module.java",
        "Mutability.java",
        "NoneType.java",
        "ParamDescriptor.java",
        "ParamTypeDescriptor.java",
        "Printer.java",
        "RangeList.java",
        "SelectorList.java",
        "SelectorValue.java",
        "Sequence.java",
        "SkylarkClassObject.java",
        "SkylarkIndexable.java",
        "SkylarkQueryable.java",
        "SkylarkType.java",
        "SkylarkUtils.java",
        "Starlark.java",
        "StarlarkCallable.java",
        "StarlarkFunction.java",
        "StarlarkIterable.java",
        "StarlarkList.java",
        "StarlarkMutable.java",
        "StarlarkThread.java",
        "StarlarkValue.java",
        "StringModule.java",
        "Tuple.java",
    ],
    deps = [
        ":frontend",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

# TODO(adonovan): merge this into :frontend and eliminate.
# Requires that lib.skylarkinterface depend on :frontend.
java_library(
    name = "semantics",
    srcs = ["StarlarkSemantics.java"],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)
