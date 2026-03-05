# Bazel's Starlark interpreter

load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
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
        "Location.java",
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
        "SyntaxError.java",
        "SyntaxUtils.java",
        "TokenKind.java",
        "TypeAliasStatement.java",
        "TypeApplication.java",
        "TypeChecker.java",
        "TypeConstructor.java",
        "TypeTagger.java",
        "Types.java",
        "UnaryOperatorExpression.java",
        "VarStatement.java",
    ],
    visibility = ["//src/main/java/net/starlark/java:clients"],
    # Do not add Bazel or Google dependencies here!
    deps = [
        "//src/main/java/net/starlark/java/spelling",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
