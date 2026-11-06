# Bazel's Starlark interpreter

load("@rules_java//java:defs.bzl", "java_library")

package(default_applicable_licenses = ["//:license"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/net/starlark/java:__pkg__"],
)

# The Starlark frontend (syntax, scanner, parser, resolver)
java_library(
    name = "syntax",
    srcs = [
        "Argument.java",
        "AssignmentStatement.java",
        "BinaryOperatorExpression.java",
        "CallExpression.java",
        "CastExpression.java",
        "Comment.java",
        "Comprehension.java",
        "ConditionalExpression.java",
        "DefStatement.java",
        "DictExpression.java",
        "DocComments.java",
        "DotExpression.java",
        "Ellipsis.java",
        "Expression.java",
        "ExpressionStatement.java",
        "FileLocations.java",
        "FileOptions.java",
        "FloatLiteral.java",
        "FlowStatement.java",
        "ForStatement.java",
        "Identifier.java",
        "IfStatement.java",
        "IndexExpression.java",
        "IntLiteral.java",
        "IsInstanceExpression.java",
        "LambdaExpression.java",
        "Lexer.java",
        "ListExpression.java",
        "LoadStatement.java",
        "Node.java",
        "NodePrinter.java",
        "NodeVisitor.java",
        "Parameter.java",
        "Parser.java",
        "ParserInput.java",
        "Program.java",
        "Resolver.java",
        "ReturnStatement.java",
        "SliceExpression.java",
        "StarlarkFile.java",
        "StarlarkType.java",
        "Statement.java",
        "StringLiteral.java",
        "SyntaxUtils.java",
        "TokenKind.java",
        "TypeAliasStatement.java",
        "TypeApplication.java",
        "TypeChecker.java",
        "TypeConstructor.java",
        "TypeContext.java",
        "TypeTable.java",
        "TypeTagger.java",
        "Types.java",
        "UnaryOperatorExpression.java",
        "VarStatement.java",
    ],
    visibility = ["//src/main/java/net/starlark/java:clients"],
    exports = [
        ":location",
        ":syntax_error",
    ],
    # Do not add Bazel or Google dependencies here!
    deps = [
        ":location",
        ":syntax_error",
        "//src/main/java/net/starlark/java/spelling",
        "//third_party/java/auto:auto_value",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:primitives",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "location",
    srcs = ["Location.java"],
    visibility = ["//src/main/java/net/starlark/java:clients"],
    # Do not add Bazel or Google dependencies here!
    deps = [
        "//third_party/java/guava:base",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "syntax_error",
    srcs = ["SyntaxError.java"],
    visibility = ["//src/main/java/net/starlark/java:clients"],
    # Do not add Bazel or Google dependencies here!
    deps = [
        ":location",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
    ],
)
