# This should be in the framework
def test(impl):
    return rule(impl, attrs = {
        'targets' : attr.label_list() # Should rather be a label_dict
        }
    )

def equal_sets(s1, s2):
    return sorted(s1) == sorted(s2)

def assert_true(value):
    if not value:
        fail("Assert failed")

#
#  Tests proper
#
load("//examples/skylark-testing/rule:my_rule.bzl", "my_rule")


# Test one dependency
def _test_my_rule_provider_one_dependency(ctx):
    # Testing logic goes here.
    t = ctx.attr.targets[0]
    d = ctx.attr.targets[1]
    assert_true(equal_sets(t.my_rule_provider, d.files))

test_my_rule_prover_one_dependency_rule = test(_test_my_rule_provider_one_dependency)

def test_my_rule_provider_one_dependency():
    # Project setup goes here:
    native.java_library(name = "dep")
    my_rule(name = "t", deps = [":dep"])
    # Actual test target
    test_my_rule_prover_one_dependency_rule(name = "test_my_rule_provider_one_dependency",
        targets = [":t", ":dep"]
    )

# Test two dependencies
def _test_my_rule_provider_two_dependencies(ctx):
    t = ctx.attr.targets[0]
    d1 = ctx.attr.targets[1]
    d2 = ctx.attr.targets[2]
    assert_true(equal_sets(t.my_rule_provider, set(d1.files) | set(d2.files)))

test_my_rule_prover_two_dependencies_rule = test(_test_my_rule_provider_two_dependencies)

def test_my_rule_provider_two_dependencies():
    native.java_library(name = "dep1")
    native.java_library(name = "dep2")
    my_rule(name = "t1", deps = [":dep1", ":dep2"])
    test_my_rule_prover_two_dependencies_rule(name = "test_my_rule_provider_two_dependencies",
        targets = [":t1", ":dep1", ":dep2"]
    )


def test_suite():
    test_my_rule_provider_one_dependency()
    test_my_rule_provider_two_dependencies()
