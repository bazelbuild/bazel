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
providers. It is currently the best option for tests that need to access the
inner workings of rules.

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

* It cannot easily test for expected failures.

The basic principle is to define a testing rule that depends on the
rule-under-test. This gives the testing rule access to the rule-under-test’s
providers. There is experimental support for passing along action information
in the form of an additional provider.

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
load("@bazel_skylib//:lib.bzl", "asserts", "unittest")
load(":myrules.bzl", "myrule", "MyInfo")

# ==== Check the provider contents ====

def _provider_contents_test_impl(ctx):
  # Analysis-time test logic; place assertions here. Always begins with begin()
  # and ends with end(). If you forget to call end(), you will get an error
  # about the test result file not having a generating action.
  env = unittest.begin(ctx)
  asserts.equals(env, "some value", ctx.attr.dep[MyInfo].val)
  # You can also use keyword arguments for readability if you prefer.
  asserts.equals(env,
    expected="some value",
    actual=ctx.attr.dep[MyInfo].val)
  return unittest.end(env)

# Create the testing rule to wrap the test logic. Note that this must be bound
# to a global variable due to restrictions on how rules can be defined. Also,
# its name must end with "_test".
provider_contents_test = unittest.make(_provider_contents_test_impl,
                                       attrs={"dep": attr.label()})
# You can use a different attrs dict if you need to take in multiple rules for
# the same unit test, or if you need to test an aspect, or if you want to
# parameterize the assertions with different expected results.

# Macro to setup the test.
def test_provider_contents():
  # Rule under test.
  myrule(name = "provider_contents_subject")
  # Testing rule.
  provider_contents_test(name = "provider_contents",
                         dep = ":provider_contents_subject")


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
  rule via `unittest.make()`, and 3) a loading-time function (macro) for
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

### Actions example

To check that the `ctx.actions.write()` line works correctly, the above example
is modified as follows.

`//mypkg/myrules.bzl`:

```python
...

myrule = rule(
    implementation = _myrule_impl,
    outputs = {"out": "%{name}.out"},
    # This enables the Actions provider for this rule.
    _skylark_testable = True,
)
```

`//mypkg/myrules_test.bzl`:

```python
...

# ==== Check the emitted file_action ====

def _file_action_test_impl(ctx):
  env = unittest.begin(ctx)
  dep = ctx.attr.dep
  # Retrieve the Actions provider.
  actions = dep[Actions]
  # Retrieve the generating action for the output file.
  action = actions.by_file[dep.out]
  # Check the content that is to be written by the action.
  asserts.equals(env, action.content, "abc")
  return unittest.end(env)

file_action_test = unittest.make(_file_action_test_impl,
                                 attrs={"dep": attr.label()})

def test_file_action():
  myrule(name = "file_action_subject")
  file_action_test(name = "file_action",
                   dep = ":file_action_subject")

...

def myrules_test_suite():
  # Call all test functions and wrap their targets in a suite.
  test_provider_contents()
  test_file_action()
  # ...

  native.test_suite(
      name = "myrules_test",
      tests = [
          ":provider_contents",
          ":file_action",
          # ...
      ]
),
```

The flag `"_skylark_testable = True"` is needed on any rule whose actions are to
be tested. This triggers the creation of the `Actions` provider. (The leading
underscore is because this API is still experimental.) The test logic for
actions makes use of the following API.

### Actions API

The [`Actions`](lib/globals.html#Actions) provider is retrieved like any other
(non-legacy) provider:

```python
ctx.attr.foo[Actions]
```

The returned object has a single field, `by_file`, which holds a dictionary
mapping each of the rule’s output files to its generating action. (Actions that
do not have output files, in particular those generated by
`ctx.actions.do_nothing()`, cannot be retrieved.)

The interface of the actions stored in the `by_file` map is documented
[here](lib/Action.html).

Finally, there is support for testing helper functions that are not rules, but
that take in a rule’s `ctx` in order to create actions on it. Use
`ctx.created_actions()` to get an `Actions` provider that has information about
all actions created on `ctx` up to the point that this function was called. For
this to work, the testing rule itself must have `"_skylark_testable=True"` set.
Testing rules created using `unittest.make()` automatically have this flag set.

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

The same framework that was used to test rules can also be used to test utility
functions (i.e., functions that are neither macros nor rule implementations).
There is no need to pass an `attrs` argument to `unittest.make()`, and there is
no special loading-time setup code to instantiate any rules-under-test. The
convenience function `unittest.suite()` can be used to reduce boilerplate in
this case.

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
load("@bazel_skylib//:lib.bzl", "asserts", "unittest")
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

This can also be used when the utility function takes in a rule’s `ctx` object
as a parameter. If the behavior of the utility function requires that the rule
be defined in a certain way, you may have to pass in an `attrs` parameter to
`unittest.make()` after all, or you may have to declare the rule manually using
`rule()`. To test helpers that create actions, make the unit test rule set
`"_skylark_testable=True"` (if it is not created via `unittest.make()`) and
write assertions on the result of `ctx.created_actions()`, as described above.
