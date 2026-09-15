# Bazel Release Documentation

This directory contains scripts to build the versioned documentation for a new Bazel release.

## Build release documentation

You can build the release documentation by running this command from within a release branch:

```
bazel build //scripts/docs:gen_release_docs --config=docs
```

This is only necessary for testing, though. There is a separate pipeline that handles this task for actual Bazel releases.

## Test scripts

You can test some of the scripts by running the following command:

```
bazel test --test_output=streamed //scripts/docs:rewriter_test
```
