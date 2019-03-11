---
layout: documentation
title: Testing
---

# Testing

There are several different approaches to testing Starlark code in Bazel. This
page gathers the current best practices and frameworks by use case.

* ToC
{:toc}

## For testing rules

[Skylib](https://github.com/bazelbuild/bazel-skylib) has a test framework called
[`unittest.bzl`](https://github.com/bazelbuild/bazel-skylib/blob/master/lib/unittest.bzl)
for checking the analysis-time behavior of rules, such as their actions and
providers. Such tests are called "analysis tests" and are currently the best
option for testing the inner workings of rules.

Some caveats:

* Test assertions occur within the build, not a separate test runner process.
  Targets that are created by the test must be named such that they do not
  collide with targets from other tests or from the build. An error that occurs
  during the test is seen by Bazel as a build breakage rather than a test
  failure.

* It requires a fair amount of boilerplate to set up the rules under test and
  the rules containing test assertions. This boilerplate may seem daunting at
  first. It helps to [keep in mind](concepts.md#evaluation-model) which code
  runs during the loading phase and which code runs during the analysis phase.

* Analysis tests are intended to be fairly small and lightweight. Certain
  features of the analysis testing framework are restricted to verifying
  targets with a maximum number of transitive dependencies (currently 500).
  This is due to performance implications of using these features with larger
  tests.

The basic principle is to define a testing rule that depends on the
rule-under-test. This gives the testing rule access to the rule-under-test’s
providers.

The testing rule’s implementation function carries out assertions. If there are
any failures, these are not raised immediately by calling `fail()` (which would
trigger an analysis-time build error), but rather by storing the errors in a
generated script that fails at test execution time.

See below for a minimal toy example, followed by an example that checks actions.

### Minimal example

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
myrules_test_suite()
```

`//mypkg/myrules.bzl`:

```python
MyInfo = provider()

def _myrule_impl(ctx):
  """Rule that just generates a file and returns a provider."""
  ctx.actions.write(ctx.outputs.out, "abc")
  return [MyInfo(val="some value", out=ctx.outputs.out)]

myrule = rule(
    implementation = _myrule_impl,
    outputs = {"out": "%{name}.out"},
)
```

`//mypkg/myrules_test.bzl`:


```python
load("@bazel_skylib//lib:unittest.bzl", "asserts", "analysistest")
load(":myrules.bzl", "myrule", "MyInfo")

# ==== Check the provider contents ====

def _provider_contents_test_impl(ctx):
  # Analysis-time test logic; place assertions here. Always begins with begin()
  # and ends with returning end(). If you forget to return end(), you will get an
  # error about an analysis test needing to return an instance of AnalysisTestResultInfo.
  env = analysistest.begin(ctx)
  target_under_test = analysistest.target_under_test(env)
  asserts.equals(env, "some value", target_under_test[MyInfo].val)
  # You can also use keyword arguments for readability if you prefer.
  asserts.equals(env,
      expected="some value",
      actual=target_under_test[MyInfo].val)
  return analysistest.end(env)

# Create the testing rule to wrap the test logic. Note that this must be bound
# to a global variable due to restrictions on how rules can be defined. Also,
# its name must end with "_test".
provider_contents_test = analysistest.make(_provider_contents_test_impl)

# Macro to setup the test.
def test_provider_contents():
  # Rule under test.
  myrule(name = "provider_contents_subject")
  # Testing rule.
  provider_contents_test(name = "provider_contents",
                         target_under_test = ":provider_contents_subject")
  # Note the target_under_test attribute is how the test rule depends on
  # the real rule target.

# Entry point from the BUILD file; macro for running each test case's macro and
# declaring a test suite that wraps them together.
def myrules_test_suite():
  # Call all test functions and wrap their targets in a suite.
  test_provider_contents()
  # ...

  native.test_suite(
      name = "myrules_test",
      tests = [
          ":provider_contents",
          # ...
      ],
  )
```

The test can be run with `bazel test //mypkg:myrules_test`.

Aside from the initial `load()` statements, there are two main parts to the
file:

* The tests themselves, each of which consists of 1) an analysis-time
  implementation function for the testing rule, 2) a declaration of the testing
  rule via `analysistest.make()`, and 3) a loading-time function (macro) for
  declaring the rule-under-test (and its dependencies) and testing rule. If the
  assertions do not change between test cases, 1) and 2) may be shared by
  multiple test cases.

* The test suite function, which calls the loading-time functions for each test,
  and declares a `test_suite` target bundling all tests together.

We recommend the following naming convention. Let `foo` stand for the part of
the test name that describes what the test is checking (`provider_contents` in
the above example). For example, a JUnit test method would be named `testFoo`.
Then:

* the loading-time function should should be named `test_foo`
  (`test_provider_contents`)

* its testing rule type should be named `foo_test` (`provider_contents_test`)

* the label of the target of this rule type should be `foo`
  (`provider_contents`)

* the implementation function for the testing rule should be named
  `_foo_test_impl` (`_provider_contents_test_impl`)

* the labels of the targets of the rules under test and their dependencies
  should be prefixed with `foo_` (`provider_contents_`)

Note that the labels of all targets can conflict with other labels in the same
BUILD package, so it’s helpful to use a unique name for the test.

### Failure Testing

It may be useful to verify that a rule fails given certain inputs or in
certain state. This can be done using the analysis test framework:

Firstly, the test rule created with `analysistest.make` should specify `expect_failure`:

```python
failure_testing_test = analysistest.make(
    _failure_testing_test_impl,
    expect_failure = True,
)
```

Secondly, the test rule implementation should make assertions on the nature
of the failure that took place (specifically, the failure message):

```python
def _failure_testing_test_impl(ctx):
    env = analysistest.begin(ctx)

    asserts.expect_failure(env, "This rule should never work")

    return analysistest.end(env)
```

### Verifying Registered Actions

You may want to write tests which make assertions about the actions that your
rule registers, for example, using `ctx.actions.run()`. This can be done in
your analysis test rule implementation function. An example:

```python
def _inspect_actions_test_impl(ctx):
    env = analysistest.begin(ctx)

    actions = analysistest.target_actions(env)
    asserts.equals(env, 1, len(actions))
    action_output = actions[0].outputs.to_list()[0]
    asserts.equals(env, "out.txt", action_output.basename)
    return analysistest.end(env)
```

Note that `analysistest.target_actions(env)` returns a list of
[`Action`](lib/Action.html) objects which represent actions registered by the
target under test.

### Verifying Rule Behavior Under Different Flags

You may want to verify your real rule behaves a certain way given certain
build flags. For example, your rule may behave differently if a user specifies:

```python
bazel build //mypkg:real_target -c opt
```
versus
```python
bazel build //mypkg:real_target -c dbg
```

At first glance, this could be done by testing the target under test using
the desired build flags:
```python
bazel test //mypkg:myrules_test -c opt
```

But then it becomes impossible for your test suite to simultaneously contain a
test which verifies the rule behavior under `-c opt` and another test which
verifies the rule behavior under `-c dbg`. Both tests would not be able to run
in the same build!

This can be solved by specifying the desired build flags when defining
the test rule:

```python
myrule_c_opt_test = analysistest.make(
    _myrule_c_opt_test_impl,
    config_settings = {
        "//command_line_option:c": "opt",
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

## For validating artifacts

There are two main ways of checking that your generated files are correct: You
can write a test script in shell, Python, or another language, and create a
target of the appropriate `*_test` rule type; or you can use a specialized rule
for the kind of test you want to perform.

### Using a test target

The most straightforward way to validate an artifact is to write a script and
add a `*_test` target to your BUILD file. The specific artifacts you want to
check should be data dependencies of this target. If your validation logic is
reusable for multiple tests, it should be a script that takes command line
arguments that are controlled by the test target’s `args` attribute. Here’s an
example that validates that the output of `myrule` from above is `"abc"`.

`//mypkg/myrule_validator.sh`:

```bash
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

### Using a custom rule

A more complicated alternative is to write the shell script as a template that
gets instantiated by a new rule. This involves more indirection and Starlark
logic, but leads to cleaner BUILD files. As a side-benefit, any argument
preprocessing can be done in Starlark instead of the script, and the script is
slightly more self-documenting since it uses symbolic placeholders (for
substitutions) instead of numeric ones (for arguments).

`//mypkg/myrule_validator.sh.template`:

```bash
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
  attrs = {"target": attr.label(single_file=True),
           # We need an implicit dependency in order to access the template.
           # A target could potentially override this attribute to modify
           # the test logic.
           "_script": attr.label(single_file=True,
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

# Needed for each target whose artifacts are to be checked. Notice that we no
# longer have to specify the output file name in a data attribute, or its
# $(location) expansion in an args attribute, or the label for the script
# (unless we want to override it).
myrule_validation_test(
    name = "validate_mytarget",
    target = ":mytarget",
)
```

Alternatively, instead of using a template expansion action, we could have
inlined the template into the .bzl file as a string and expanded it during the
analysis phase using the `str.format` method or `%`-formatting.


## For testing Starlark utilities

[Skylib](https://github.com/bazelbuild/bazel-skylib)'s
[`unittest.bzl`](https://github.com/bazelbuild/bazel-skylib/blob/master/lib/unittest.bzl)
framework can be used to test utility functions (that is, functions that are neither
macros nor rule implementations). Instead of using `unittest.bzl`'s `analysistest`
library, `unittest` may be used.
For such test suites, the convenience function `unittest.suite()` can be used to
reduce boilerplate.

`//mypkg/BUILD`:

```python
load(":myhelpers_test.bzl", "myhelpers_test_suite")

myhelpers_test_suite()
```

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

def myhelpers_test_suite():
  # unittest.suite() takes care of instantiating the testing rules and creating
  # a test_suite.
  unittest.suite(
    "myhelpers_tests",
    myhelper_test,
    # ...
  )
```

For more examples, see Skylib’s own [tests](https://github.com/bazelbuild/bazel-skylib/blob/master/tests/BUILD).

