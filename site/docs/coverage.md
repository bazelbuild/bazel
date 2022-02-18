---
layout: documentation
title: Code Coverage with Bazel
---

<div style="background-color: #EFCBCB; color: #AE2B2B;  border: 1px solid #AE2B2B; border-radius: 5px; border-left: 10px solid #AE2B2B; padding: 0.5em;">
<b>IMPORTANT:</b> The Bazel docs have moved! Please update your bookmark to <a href="https://bazel.build/docs/coverage" style="color: #0000EE;">https://bazel.build/docs/coverage</a>
<p/>
You can <a href="https://blog.bazel.build/2022/02/17/Launching-new-Bazel-site.html" style="color: #0000EE;">read about</a> the migration, and let us <a href="https://forms.gle/onkAkr2ZwBmcbWXj7" style="color: #0000EE;">know what you think</a>.
</div>


# Code coverage with Bazel

Bazel features a `coverage` sub-command to produce code coverage
reports on repositories that can be tested with `bazel coverage`. Due
to the idiosyncrasies of the various language ecosystems, it is not
always trivial to make this work for a given project.

This page documents the general process for creating and viewing
coverage reports, and also features some language-specific notes for
languages whose configuration is well-known. It is best read by first
reading [the general section](#Creating-a-coverage-report), and then
reading about the requirements for a specific language. Note also the
[remote execution section](#Remote-execution), which requires some
additional considerations.

While a lot of customization is possible, this document focuses on
producing and consuming [`lcov`][lcov] reports, which is currently the
most well-supported route.

## Creating a coverage report

### Preparation

The basic workflow for creating coverage reports requires the
following:

- A basic repository with test targets
- A toolchain with the language-specific code coverage tools installed
- A correct "instrumentation" configuration

The former two are language-specific and mostly straightforward,
however the latter can be more difficult for complex projects.

"Instrumentation" in this case refers to the coverage tools that are
used for a specific target. Bazel allows turning this on for a
specific subset of files using the
[`--instrumentation_filter`](../command-line-reference.html#flag--instrumentation_filter)
flag, which specifies a filter for targets that are tested with the
instrumentation enabled. To enable instrumentation for tests, the
[`--instrument_test_targets`](../command-line-reference.html#flag--instrument_test_targets)
flag is required.

By default, bazel tries to match the target package(s), and prints the
relevant filter as an `INFO` message.

### Running coverage

To produce a coverage report, use [`bazel coverage
--combined_report=lcov
[target]`](../command-line-reference.html#coverage). This runs the
tests for the target, generating coverage reports in the lcov format
for each file.

Once finished, bazel runs an action that collects all the produced
coverage files, and merges them into one, which is then finally
created under `$(bazel info
output_path)/_coverage/_coverage_report.dat`.

Coverage reports are also produced if tests fail, though note that
this does not extend to the failed tests - only passing tests are
reported.

### Viewing coverage

The coverage report is only output in the non-human-readable `lcov`
format. From this, we can use the `genhtml` utility (part of [the lcov
project][lcov]) to produce a report that can be viewed in a web
browser:

```console
genhtml --output genhtml "$(bazel info output_path)/_coverage/_coverage_report.dat"
```

Note that `genhtml` reads the source code as well, to annotate missing
coverage in these files. For this to work, it is expected that
`genhtml` is executed in the root of the bazel project.

To view the result, simply open the `index.html` file produced in the
`genhtml` directory in any web browser.

For further help and information around the `genhtml` tool, or the
`lcov` coverage format, see [the lcov project][lcov].

## Remote execution

Running with remote test execution currently has a few caveats:

- The report combination action cannot yet run remotely. This is
  because Bazel does not consider the coverage output files as part of
  its graph (see [this issue][remote_report_issue]), and can therefore
  not correctly treat them as inputs to the combination action. To
  work around this, use `--strategy=CoverageReport=local`.
  - Note: It may be necessary to specify something like
    `--strategy=CoverageReport=local,remote` instead, if Bazel is set
    up to try `local,remote`, due to how Bazel resolves strategies.
- `--remote_download_minimal` and similar flags can also not be used
  as a consequence of the former.
- Bazel will currently fail to create coverage information if tests
  have been cached previously. To work around this,
  `--nocache_test_results` can be set specifically for coverage runs,
  although this of course incurs a heavy cost in terms of test times.
- `--experimental_split_coverage_postprocessing` and
  `--experimental_fetch_all_coverage_outputs`
  - Usually coverage is run as part of the test action, and so by
    default, we don't get all coverage back as outputs of the remote
    execution by default. These flags override the default and obtain
    the coverage data. See [this issue][split_coverage_issue] for more
    details.

## Language-specific configuration

### Java

Java should work out-of-the-box with the default configuration. The
[bazel toolchains][bazel_toolchains] contain everything necessary for
remote execution, as well, including JUnit.

### Python

#### Prerequisites

Running coverage with python has some prerequisites:

- A bazel binary that includes [b01c859][python_coverage_commit],
  which should be any Bazel >3.0.
- A [modified version of coverage.py][modified_coveragepy].
<!-- TODO: Upstream an lcov implementation so that this becomes usable  -->

#### Consuming the modified coverage.py

A way to do this is via [rules_python][rules_python], this provides
the ability to use a `requirements.txt` file, the requirements listed
in the file are then created as bazel targets using the
[pip_install][pip_install_rule] repository rule.

The `requirements.txt` should have the following entry:

```text
git+https://github.com/ulfjack/coveragepy.git@lcov-support
```

The `rules_python`, `pip_install`, and the `requirements.txt` file should then be used in the WORKSPACE file as:

```python
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.5.0/rules_python-0.5.0.tar.gz",
    sha256 = "cd6730ed53a002c56ce4e2f396ba3b3be262fd7cb68339f0377a45e8227fe332",
)

load("@rules_python//python:pip.bzl", "pip_install")

pip_install(
   name = "python_deps",
   requirements = "//:requirements.txt",
)
```

Note: The version of `rules_python` is incidental - this was simply
the latest at the time of writing. Refer to the
[upstream][rules_python] for up-to-date instructions.

The coverage.py requirement can then be consumed by test targets by
setting the following in `BUILD` files:

```python
load("@python_deps//:requirements.bzl", "entry_point")

alias(
    name = "python_coverage_tools",
    actual = entry_point("coverage"),
)

py_test(
    name = "test",
    srcs = ["test.py"],
    env = {
        "PYTHON_COVERAGE": "$(location :python_coverage_tools)",
    },
    deps = [
        ":main",
        ":python_coverage_tools",
    ],
)
```
<!-- TODO: Allow specifying a target for `PYTHON_COVERAGE`, instead of having to use `$(location)` -->


[lcov]: https://github.com/linux-test-project/lcov
[rules_python]: https://github.com/bazelbuild/rules_python
[bazel_toolchains]: https://github.com/bazelbuild/bazel-toolchains
[remote_report_issue]: https://github.com/bazelbuild/bazel/issues/4685
[split_coverage_issue]: https://github.com/bazelbuild/bazel/issues/4685
[python_coverage_commit]: https://github.com/bazelbuild/bazel/commit/b01c85962d88661ec9f6c6704c47d8ce67ca4d2a
[modified_coveragepy]: https://github.com/ulfjack/coveragepy/tree/lcov-support
[pip_install_rule]: https://github.com/bazelbuild/rules_python#installing-pip-dependencies
