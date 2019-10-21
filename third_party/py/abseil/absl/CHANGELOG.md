# Python Absl Changelog

All notable changes to Python Absl are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com).

## Unreleased

Nothing notable unreleased.

## 0.8.1 (2019-10-08)

### Fixed

*   (testing) `absl.testing`'s pretty print reporter no longer buffers
    RUN/OK/FAILED messages.
*   (testing) `create_tempfile` will overwrite pre-existing read-only files.

## 0.8.0 (2019-08-26)

### Added

*   (testing) `absltest.expectedFailureIf`: a variant of
    `unittest.expectedFailure` that allows a condition to be given.

### Changed

*   (bazel) Tests now pass when bazel
    `--incompatible_allow_python_version_transitions=true` is set.
*   (bazel) Both Python 2 and Python 3 versions of tests are now created. To
    only run one major Python version, use
    `bazel test --test_tag_filters=-python[23]` to ignore the other version.
*   (testing) `assertTotallyOrdered` no longer requires objects to implement
    `__hash__`.
*   (testing) `absltest` now integrates better with `--pdb_post_mortem`.
*   (testing) `xml_reporter` now includes timestamps to testcases, test_suite,
    test_suites elements.

### Fixed

*    #99: `absl.logging` no longer registers itself to `logging.root` at import
     time.
*    #108: Tests now pass with Bazel 0.28.0 on macOS.

## 0.7.1 (2019-03-12)

### Added

*   (flags) `flags.mark_bool_flags_as_mutual_exclusive`: convenience function to
    check that only one, or at most one, flag among a set of boolean flags are
    True.

### Changed

*   (bazel) Bazel 0.23+ or 0.22+ is now required for building/testing.
    Specifically, a Bazel version that supports
    `@bazel_tools//tools/python:python_version` for selecting the Python
    version.

### Fixed

*   #94: LICENSE files are now included in sdist.
*   #93: Change log added.

## 0.7.0 (2019-01-11)

### Added

*   (bazel) testonly=1 has been removed from the testing libraries, which allows
    their use outside of testing contexts.
*   (flags) Multi-flags now accept any Iterable type for the default value
    instead of only lists. Strings are still special cased as before. This
    allows sets, generators, views, etc to be used naturally.
*   (flags) DEFINE_multi_enum_class: a multi flag variant of enum_class.
*   (testing) Most of absltest is now type-annotated.
*   (testing) Made AbslTest.assertRegex available under Python 2. This allows
    Python 2 code to write more natural Python 3 compatible code. (Note: this
    was actually released in 0.6.1, but unannounced)
*   (logging) logging.vlog_is_on: helper to tell if a vlog() call will actually
    log anything. This allows avoiding computing expansive inputs to a logging
    call when logging isn't enabled for that level.

### Fixed

*   (flags) Pickling flags now raises an clear error instead of a cryptic one.
    Pickling flags isn't supported; instead use flags_into_string to serialize
    flags.
*   (flags) Flags serialization works better: the resulting serialized value,
    when deserialized, won't cause --help to be invoked, thus ending the
    process.
*   (flags) Several flag fixes to make them behave more like the Absl C++ flags:
    empty --flagfile is allowed; --nohelp and --help=false don't display help
*   (flags) An empty --flagfile value (e.g. "--flagfile=" or "--flagfile=''"
    doesn't raise an error; its not just ignored. This matches Abseil C++
    behavior.
*   (bazel) Building with Bazel 0.2.0 works without extra incompatiblity disable
    build flags.

### Changed

*   (flags) Flag serialization is now deterministic: this improves Bazel build
    caching for tools that are affected by flag serialization.

## 0.6.0 (2018-10-22)

### Added

*   Tempfile management APIs for tests: read/write/manage tempfiles for test
    purposes easily and correctly. See TestCase.create_temp{file/dir} and the
    corresponding commit for more info.

## 0.5.0 (2018-09-17)

### Added

*   Flags enum support: flags.DEFINE_enum_class allows using an `Enum` derived
    class to define the allowed values for a flag.

## 0.4.1 (2018-08-28)

### Fixed

*   Flags no long allow spaces in their names

### Changed

*   XML test output is written at the end of all test execution.
*   If the current user's username can't be gotten, fallback to uid, else fall
    back to a generic 'unknown' string.

## 0.4.0 (2018-08-14)

### Added

*   argparse integration: absl-registered flags can now be accessed via argparse
    using absl.flags.argparse_flags: see that module for more information.
*   TestCase.assertSameStructure now allows mixed set types.

### Changed

*   Test output now includes start/end markers for each test ran. This is to
    help distinguish output from tests clearly.

## 0.3.0 (2018-07-25)

### Added

*   `app.call_after_init`: Register functions to be called after app.run() is
    called. Useful for program-wide initialization that library code may need.
*   `logging.log_every_n_seconds`: like log_every_n, but based on elapsed time
    between logging calls.
*   `absltest.mock`: alias to unittest.mock (PY3) for better unittest drop-in
    replacement. For PY2, it will be available if mock is importable.

### Fixed

*   `ABSLLogger.findCaller()`: allow stack_info arg and return value for PY2
*   Make stopTest locking reentrant: this prevents deadlocks for test frameworks
    that customize unittest.TextTestResult.stopTest.
*   Make --helpfull work with unicode flag help strings.
