Project: /_project.yaml
Book: /_book.yaml

# Testing

{% include "_buttons.html" %}

There are several different approaches to testing Starlark code in Bazel. This
page gathers the current best practices and frameworks by use case.

## Testing rules {:#testing-rules}

[Skylib](https://github.com/bazelbuild/bazel-skylib){: .external} has a test framework called
[`unittest.bzl`](https://github.com/bazelbuild/bazel-skylib/blob/main/lib/unittest.bzl){: .external}
for checking the analysis-time behavior of rules, such as their actions and
providers. Such tests are called "analysis tests" and are currently the best
option for testing the inner workings of rules.

Some caveats:

*   Test assertions occur within the build, not a separate test runner process.
    Targets that are created by the test must be named such that they do not
    collide with targets from other tests or from the build. An error that
    occurs during the test is seen by Bazel as a build breakage rather than a
    test failure.

*   It requires a fair amount of boilerplate to set up the rules under test and
    the rules containing test assertions. This boilerplate may seem daunting at
    first. It helps to [keep in mind](/extending/concepts#evaluation-model) that macros
    are evaluated and targets generated during the loading phase, while rule
    implementation functions don't run until later, during the analysis phase.

*   Analysis tests are intended to be fairly small and lightweight. Certain
    features of the analysis testing framework are restricted to verifying
    targets with a maximum number of transitive dependencies (currently 500).
    This is due to performance implications of using these features with larger
    tests.

The basic principle is to define a testing rule that depends on the
rule-under-test. This gives the testing rule access to the rule-under-test's
providers.

The testing rule's implementation function carries out assertions. If there are
any failures, these are not raised immediately by calling `fail()` (which would
trigger an analysis-time build error), but rather by storing the errors in a
generated script that fails at test execution time.

See below for a minimal toy example, followed by an example that checks actions.

### Minimal example {:#testing-rules-example}

`//mypkg/myrules.bzl`:

```python
MyInfo = provider(fields = {
    "val": "string value",
    "out": "output File",
})

def _myrule_impl(ctx):
    """Rule that just generates a file and returns a provider."""
    out = ctx.actions.declare_file(ctx.label.name + ".out")
    ctx.actions.write(out, "abc")
    return [MyInfo(val="some value", out=out)]

myrule = rule(
    implementation = _myrule_impl,
)
```

`//mypkg/myrules_test.bzl`:


```python
load("@bazel_skylib//lib:unittest.bzl", "asserts", "analysistest")
load(":myrules.bzl", "myrule", "MyInfo")

# ==== Check the provider contents ====

def _provider_contents_test_impl(ctx):
    env = analysistest.begin(ctx)

    target_under_test = analysistest.target_under_test(env)
    # If preferred, could pass these values as "expected" and "actual" keyword
    # arguments.
    asserts.equals(env, "some value", target_under_test[MyInfo].val)

    # If you forget to return end(), you will get an error about an analysis
    # test needing to return an instance of AnalysisTestResultInfo.
    return analysistest.end(env)

# Create the testing rule to wrap the test logic. This must be bound to a global
# variable, not called in a macro's body, since macros get evaluated at loading
# time but the rule gets evaluated later, at analysis time. Since this is a test
# rule, its name must end with "_test".
provider_contents_test = analysistest.make(_provider_contents_test_impl)

# Macro to setup the test.
def _test_provider_contents():
    # Rule under test. Be sure to tag 'manual', as this target should not be
    # built using `:all` except as a dependency of the test.
    myrule(name = "provider_contents_subject", tags = ["manual"])
    # Testing rule.
    provider_contents_test(name = "provider_contents_test",
                           target_under_test = ":provider_contents_subject")
    # Note the target_under_test attribute is how the test rule depends on
    # the real rule target.

# Entry point from the BUILD file; macro for running each test case's macro and
# declaring a test suite that wraps them together.
def myrules_test_suite(name):
    # Call all test functions and wrap their targets in a suite.
    _test_provider_contents()
    # ...

    native.test_suite(
        name = name,
        tests = [
            ":provider_contents_test",
            # ...
        ],
    )
```

`//mypkg/BUILD`:

```python
load(":myrules.bzl", "myrule")
load(":myrules_test.bzl", "myrules_test_suite")

# Production use of the rule.
myrule(
    name = "mytarget",
)

# Call a macro that defines targets that perform the tests at analysis time,
# and that can be executed with "bazel test" to return the result.
myrules_test_suite(name = "myrules_test")
```

The test can be run with `bazel test //mypkg:myrules_test`.

Aside from the initial `load()` statements, there are two main parts to the
file:

*   The tests themselves, each of which consists of 1) an analysis-time
    implementation function for the testing rule, 2) a declaration of the
    testing rule via `analysistest.make()`, and 3) a loading-time function
    (macro) for declaring the rule-under-test (and its dependencies) and testing
    rule. If the assertions do not change between test cases, 1) and 2) may be
    shared by multiple test cases.

*   The test suite function, which calls the loading-time functions for each
    test, and declares a `test_suite` target bundling all tests together.

For consistency, follow the recommended naming convention: Let `foo` stand for
the part of the test name that describes what the test is checking
(`provider_contents` in the above example). For example, a JUnit test method
would be named `testFoo`.

Then:

*   the macro which generates the test and target under test should should be
    named `_test_foo` (`_test_provider_contents`)

*   its test rule type should be named `foo_test` (`provider_contents_test`)

*   the label of the target of this rule type should be `foo_test`
    (`provider_contents_test`)

*   the implementation function for the testing rule should be named
    `_foo_test_impl` (`_provider_contents_test_impl`)

*   the labels of the targets of the rules under test and their dependencies
    should be prefixed with `foo_` (`provider_contents_`)

Note that the labels of all targets can conflict with other labels in the same
BUILD package, so it's helpful to use a unique name for the test.

### Failure testing {:#failure-testing}

It may be useful to verify that a rule fails given certain inputs or in certain
state. This can be done using the analysis test framework:

The test rule created with `analysistest.make` should specify `expect_failure`:

```python
failure_testing_test = analysistest.make(
    _failure_testing_test_impl,
    expect_failure = True,
)
```

The test rule implementation should make assertions on the nature of the failure
that took place (specifically, the failure message):

```python
def _failure_testing_test_impl(ctx):
    env = analysistest.begin(ctx)
    asserts.expect_failure(env, "This rule should never work")
    return analysistest.end(env)
```

Also make sure that your target under test is specifically tagged 'manual'.
Without this, building all targets in your package using `:all` will result in a
build of the intentionally-failing target and will exhibit a build failure. With
'manual', your target under test will build only if explicitly specified, or as
a dependency of a non-manual target (such as your test rule):

```python
def _test_failure():
    myrule(name = "this_should_fail", tags = ["manual"])

    failure_testing_test(name = "failure_testing_test",
                         target_under_test = ":this_should_fail")

# Then call _test_failure() in the macro which generates the test suite and add
# ":failure_testing_test" to the suite's test targets.
```

### Verifying registered actions {:#verifying-registered-actions}

You may want to write tests which make assertions about the actions that your
rule registers, for example, using `ctx.actions.run()`. This can be done in your
analysis test rule implementation function. An example:

```python
def _inspect_actions_test_impl(ctx):
    env = analysistest.begin(ctx)

    target_under_test = analysistest.target_under_test(env)
    actions = analysistest.target_actions(env)
    asserts.equals(env, 1, len(actions))
    action_output = actions[0].outputs.to_list()[0]
    asserts.equals(
        env, target_under_test.label.name + ".out", action_output.basename)
    return analysistest.end(env)
```

Note that `analysistest.target_actions(env)` returns a list of
[`Action`](lib/Action) objects which represent actions registered by the
target under test.

### Verifying rule behavior under different flags {:#verifying-rule-behavior}

You may want to verify your real rule behaves a certain way given certain build
flags. For example, your rule may behave differently if a user specifies:

```shell
bazel build //mypkg:real_target -c opt
```

versus

```shell
bazel build //mypkg:real_target -c dbg
```

At first glance, this could be done by testing the target under test using the
desired build flags:

```shell
bazel test //mypkg:myrules_test -c opt
```

But then it becomes impossible for your test suite to simultaneously contain a
test which verifies the rule behavior under `-c opt` and another test which
verifies the rule behavior under `-c dbg`. Both tests would not be able to run
in the same build!

This can be solved by specifying the desired build flags when defining the test
rule:

```python
myrule_c_opt_test = analysistest.make(
    _myrule_c_opt_test_impl,
    config_settings = {
        "//command_line_option:compilation_mode": "opt",
    },
)
```

Normally, a target under test is analyzed given the current build flags.
Specifying `config_settings` overrides the values of the specified command line
options. (Any unspecified options will retain their values from the actual
command line).

In the specified `config_settings` dictionary, command line flags must be
prefixed with a special placeholder value `//command_line_option:`, as is shown
above.


## Validating artifacts {:#validating-artifacts}

The main ways to check that your generated files are correct are:

*   You can write a test script in shell, Python, or another language, and
    create a target of the appropriate `*_test` rule type.

*   You can use a specialized rule for the kind of test you want to perform.

### Using a test target {:#using-test-target}

The most straightforward way to validate an artifact is to write a script and
add a `*_test` target to your BUILD file. The specific artifacts you want to
check should be data dependencies of this target. If your validation logic is
reusable for multiple tests, it should be a script that takes command line
arguments that are controlled by the test target's `args` attribute. Here's an
example that validates that the output of `myrule` from above is `"abc"`.

`//mypkg/myrule_validator.sh`:

```shell
if [ "$(cat $1)" = "abc" ]; then
  echo "Passed"
  exit 0
else
  echo "Failed"
  exit 1
fi
```

`//mypkg/BUILD`:

```python
...

myrule(
    name = "mytarget",
)

...

# Needed for each target whose artifacts are to be checked.
sh_test(
    name = "validate_mytarget",
    srcs = [":myrule_validator.sh"],
    args = ["$(location :mytarget.out)"],
    data = [":mytarget.out"],
)
```

### Using a custom rule {:#using-custom-rule}

A more complicated alternative is to write the shell script as a template that
gets instantiated by a new rule. This involves more indirection and Starlark
logic, but leads to cleaner BUILD files. As a side-benefit, any argument
preprocessing can be done in Starlark instead of the script, and the script is
slightly more self-documenting since it uses symbolic placeholders (for
substitutions) instead of numeric ones (for arguments).

`//mypkg/myrule_validator.sh.template`:

```shell
if [ "$(cat %TARGET%)" = "abc" ]; then
  echo "Passed"
  exit 0
else
  echo "Failed"
  exit 1
fi
```

`//mypkg/myrule_validation.bzl`:

```python
def _myrule_validation_test_impl(ctx):
  """Rule for instantiating myrule_validator.sh.template for a given target."""
  exe = ctx.outputs.executable
  target = ctx.file.target
  ctx.actions.expand_template(output = exe,
                              template = ctx.file._script,
                              is_executable = True,
                              substitutions = {
                                "%TARGET%": target.short_path,
                              })
  # This is needed to make sure the output file of myrule is visible to the
  # resulting instantiated script.
  return [DefaultInfo(runfiles=ctx.runfiles(files=[target]))]

myrule_validation_test = rule(
    implementation = _myrule_validation_test_impl,
    attrs = {"target": attr.label(allow_single_file=True),
             # You need an implicit dependency in order to access the template.
             # A target could potentially override this attribute to modify
             # the test logic.
             "_script": attr.label(allow_single_file=True,
                                   default=Label("//mypkg:myrule_validator"))},
    test = True,
)
```

`//mypkg/BUILD`:

```python
...

myrule(
    name = "mytarget",
)

...

# Needed just once, to expose the template. Could have also used export_files(),
# and made the _script attribute set allow_files=True.
filegroup(
    name = "myrule_validator",
    srcs = [":myrule_validator.sh.template"],
)

# Needed for each target whose artifacts are to be checked. Notice that you no
# longer have to specify the output file name in a data attribute, or its
# $(location) expansion in an args attribute, or the label for the script
# (unless you want to override it).
myrule_validation_test(
    name = "validate_mytarget",
    target = ":mytarget",
)
```

Alternatively, instead of using a template expansion action, you could have
inlined the template into the .bzl file as a string and expanded it during the
analysis phase using the `str.format` method or `%`-formatting.

## Testing Starlark utilities {:#testing-starlark-utilities}

[Skylib](https://github.com/bazelbuild/bazel-skylib){: .external}'s
[`unittest.bzl`](https://github.com/bazelbuild/bazel-skylib/blob/main/lib/unittest.bzl){: .external}
framework can be used to test utility functions (that is, functions that are
neither macros nor rule implementations). Instead of using `unittest.bzl`'s
`analysistest` library, `unittest` may be used. For such test suites, the
convenience function `unittest.suite()` can be used to reduce boilerplate.

`//mypkg/myhelpers.bzl`:

```python
def myhelper():
    return "abc"
```

`//mypkg/myhelpers_test.bzl`:


```python
load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load(":myhelpers.bzl", "myhelper")

def _myhelper_test_impl(ctx):
  env = unittest.begin(ctx)
  asserts.equals(env, "abc", myhelper())
  return unittest.end(env)

myhelper_test = unittest.make(_myhelper_test_impl)

# No need for a test_myhelper() setup function.

def myhelpers_test_suite(name):
  # unittest.suite() takes care of instantiating the testing rules and creating
  # a test_suite.
  unittest.suite(
    name,
    myhelper_test,
    # ...
  )
```

`//mypkg/BUILD`:

```python
load(":myhelpers_test.bzl", "myhelpers_test_suite")

myhelpers_test_suite(name = "myhelpers_tests")
```

For more examples, see Skylib's own [tests](https://github.com/bazelbuild/bazel-skylib/blob/main/tests/BUILD){: .external}.
