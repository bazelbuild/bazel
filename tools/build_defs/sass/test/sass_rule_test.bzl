load(
    "/tools/build_defs/sass/sass",
    "sass_binary",
)

load(
    "/tools/build_rules/test_rules",
    "success_target",
    "successful_test",
    "failure_target",
    "failed_test",
    "assert_",
    "strip_prefix",
    "expectation_description",
    "check_results",
    "load_results",
    "analysis_results",
    "rule_test",
    "file_test",
)

def _sass_binary_test(package):
    rule_test(
        name = "hello_world_rule_test",
        generates = ["hello_world.css", "hello_world.css.map"],
        rule = package + "/hello_world:hello_world",
    )

def sass_rule_test(package):
    """Issue simple tests on sass rules."""
    _sass_binary_test(package)
