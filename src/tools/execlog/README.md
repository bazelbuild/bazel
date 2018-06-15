# Execution Log Parser

This tool is used to inspect and parse the Bazel execution logs.
To generate the execution log, run e.g.:

        bazel build \
            --experimental_execution_log_file=/tmp/exec.log :hello_world

Then build the parser and run it.

        bazel build src/tools/execlog:all
        bazel-bin/src/tools/execlog/parser --log_path=/tmp/exec.log

This will simply print the log contents to stdout in text form.
