package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_js_library")

# Not a real polyfill. Do NOT use for anything, but tests.
closure_js_library(
    name = "polyfill",
    srcs = ["polyfill.js"],
    suppress = [
        "JSC_INVALID_OPERAND_TYPE",
        "JSC_MISSING_JSDOC",
        "JSC_STRICT_INEXISTENT_PROPERTY",
        "JSC_TYPE_MISMATCH",
        "JSC_UNKNOWN_EXPR_TYPE",
    ],
)

# Do NOT use this artifact; it is for test purposes only.
closure_js_library(
    name = "decode",
    srcs = ["decode.js"],
    suppress = [
        "JSC_DUP_VAR_DECLARATION",
        "JSC_USELESS_BLOCK",
    ],
    deps = [":polyfill"],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_js_test")

closure_js_test(
    name = "all_tests",
    srcs = ["decode_test.js"],
    deps = [
        ":decode",
        ":polyfill",
        "@io_bazel_rules_closure//closure/library:testing",
    ],
)
