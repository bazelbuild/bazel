# Build Performance Benchmark

This benchmark is used to measure and compare Bazel's performance (elapsed time
of builds) between several commits. It requires bazel and git to be installed.

To run the benchmark:

1. `bazel build //src/tools/benchmark/java/com/google/devtools/build/benchmark:benchmark`
2. Run the built binary and follow the instructions.
3. Put the benchmark result file as `<name>.json` into the directory
   `src/tools/benchmark/webapp/data`.
4. Put the string `<name>.json` into the file
   `src/tools/benchmark/webapp/file_list`.
5. Start an HTTP server there and open `/index.html`.
   - Hint: You can start a simple HTTP server by running
   `python -m SimpleHTTPServer` (Python 2) or
   `python3 -m http.server` (Python 3).

A hosted version of the benchmark that is kept up to date by our CI system is
available here: https://perf.bazel.build/.
