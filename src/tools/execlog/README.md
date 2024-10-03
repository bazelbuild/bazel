# Execution Log Parser

This tool is used to inspect and parse the Bazel execution logs. Currently
supported formats are `binary`, `json`, and `compact`.

To generate the execution log, run e.g.:

        bazel build \
            --execution_log_compact_file=/tmp/exec.log :hello_world

Then build the parser and run it:

        bazel build src/tools/execlog:parser
        bazel-bin/src/tools/execlog/parser --log_path=/tmp/exec.log

This will simply print the log contents to stdout in text form.

To output results to a file, use `--output_path`:

        bazel-bin/src/tools/execlog/parser --log_path=/tmp/exec.log \
            --output_path=/tmp/exec.log.txt

To limit the output to a certain runner, use `--restrict_to_runner` option.
For example,

        bazel-bin/src/tools/execlog/parser --log_path=/tmp/exec.log \
            --restrict_to_runner="linux-sandbox"

Will limit the output to those actions that were ran in the linux sandbox.


Note that because Bazel is nondeterministic, different runs of the same build
may produce logs where actions are in a different order. To achieve a more
meaningful textual diff, use the parser to convert both files at the same time:

        bazel-bin/src/tools/execlog/parser --log_path=/tmp/exec1.log \
                                           --log_path=/tmp/exec2.log \
                                           --output_path=/tmp/exec1.log.txt \
                                           --output_path=/tmp/exec2.log.txt

This will convert `/tmp/exec1.log` to text as-is, but will reorder `/tmp/exec2.log`
to match the order of actions found in `/tmp/exec1.log`. Actions are matched if
their first output is the same. Actions that are not found in `/tmp/exec1.log`
are put at the end of `/tmp/exec2.log.txt`.

Note that this reordering makes it easier to see differences using text-based
diffing tools, but may break the logical sequence of actions in
`/tmp/exec2.log.txt`.

# Execution Log Converter

This tool is used to convert between Bazel execution log formats.

For example, to convert from the binary format to the JSON format:

        bazel build src/tools/execlog:converter
        bazel-bin/src/tools/execlog/converter \
            --input binary:/tmp/binary.log --output json:/tmp/json.log

By default, the output will be in the same order as the input. To sort in a
deterministic order, use --sort.
