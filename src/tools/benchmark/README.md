# Benchmark

This benchmark is for comparing performance (elapsed time) between several bazel commits.
It requires bazel and git installed.

To Run the benchmark:
1. `bazel build //src/tools/benchmark/java/com/google/devtools/build/benchmark:benchmark`
2. run the built binary and follow the argument instruction
3. put the benchmark result as `<name>.json` into `src/tools/benchmark/webapp/data`
4. put the string `<name>.json` into `src/tools/benchmark/webapp/file_list`
5. start an http server there and open `/index.html`
