# Test data for //src/test/py/bazel:bazel_external_repository_test


The archives here are hand built first rather than making them during the
test. This reduces complexity of the test.

hello-1.0.0.tar.gz is created with
```
tar czf hello-1.0.0.tar.gz hello-1.0.0
sha256sum hello-1.0.0.tar.gz
```

Update src/test/py/bazel/bazel_external_repository_test.py
with the new checksum
