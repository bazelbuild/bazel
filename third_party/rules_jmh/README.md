# rules_jmh

Bazel rules for generating and running microbenchmarks with [JMH](https://openjdk.java.net/projects/code-tools/jmh/).

# Usage

Load the JMH dependencies in your WORKSPACE file

```python
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
  name = "rules_jmh",
  strip_prefix = "buchgr-rules_jmh-6ccf8d7",
  url = "https://github.com/buchgr/rules_jmh/zipball/6ccf8d7b270083982e5c143935704b9f3f18b256",
  type = "zip",
  sha256 = "dbb7d7e5ec6e932eddd41b910691231ffd7b428dff1ef9a24e4a9a59c1a1762d",
)

load("@rules_jmh//:deps.bzl", "rules_jmh_deps")
rules_jmh_deps()
load("@rules_jmh//:defs.bzl", "rules_jmh_maven_deps")
rules_jmh_maven_deps()
```

You can specify JMH benchmarks by using the `jmh_java_benchmarks` rule. It takes the same arguments as `java_binary` except for `main_class`. One can specify one or more JMH benchmarks in the `srcs` attribute.


```python
load("@rules_jmh//:defs.bzl", "jmh_java_benchmarks")

jmh_java_benchmarks(
    name = "example-benchmarks",
    srcs = ["Benchmark1.java", "Benchmark2.java"]
)
```

You can run the benchmark `//:example-benchmarks` using `bazel run`.
```sh
$ bazel run :example-benchmarks
```

and also pass JMH command line flags

```sh
$ bazel run :example-benchmarks -- -f 0 -bm avgt
```

Alternatively you can also build a standalone fat (deploy) jar that contains all dependencies and can be run without Bazel

```sh
# Build the jar file
$ bazel build :example-benchmarks_deploy.jar
# Shutdown Bazel for it to not interfere with your benchmark
$ bazel shutdown
# Run the benchmark
$ java -jar bazel-bin/example-benchmarks_deploy.jar
```