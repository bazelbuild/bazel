# Execution Graph Log Parser

This tool is used to inspect and parse the Bazel execution graph log. The log is
a zstd-compressed stream of length-delimited `execution_graph.Node` protos (see
`src/main/protobuf/execution_graph.proto`).

To generate the execution graph log, run e.g.:

        bazel build \
            --experimental_enable_execution_graph_log \
            --experimental_execution_graph_log_path=/tmp/exec_graph.log :hello_world

Then build the parser and run it:

        bazel build src/tools/execgraph:parser
        bazel-bin/src/tools/execgraph/parser --log_path=/tmp/exec_graph.log

This will simply print the log contents to stdout in text form.

To output results to a file, use `--output_path`:

        bazel-bin/src/tools/execgraph/parser --log_path=/tmp/exec_graph.log \
            --output_path=/tmp/exec_graph.log.txt

To limit the output to a certain runner, use `--restrict_to_runner` option.
For example,

        bazel-bin/src/tools/execgraph/parser --log_path=/tmp/exec_graph.log \
            --restrict_to_runner="linux-sandbox"

Will limit the output to those nodes that were ran in the linux sandbox.


Note that because Bazel is nondeterministic, different runs of the same build
may produce logs where nodes are in a different order. To achieve a more
meaningful textual diff, use the parser to convert both files at the same time:

        bazel-bin/src/tools/execgraph/parser --log_path=/tmp/exec_graph1.log \
                                             --log_path=/tmp/exec_graph2.log \
                                             --output_path=/tmp/exec_graph1.log.txt \
                                             --output_path=/tmp/exec_graph2.log.txt

This will convert `/tmp/exec_graph1.log` to text as-is, but will reorder
`/tmp/exec_graph2.log` to match the order of nodes found in
`/tmp/exec_graph1.log`. Nodes are matched if their description is the same.
Nodes that are not found in `/tmp/exec_graph1.log` are put at the end of
`/tmp/exec_graph2.log.txt`. Note that descriptions are not guaranteed to be
unique -- a single action can emit several nodes with the same description (for
example, retries or actions that issue multiple spawns) -- so matching is
best-effort when descriptions repeat.

Note that this reordering makes it easier to see differences using text-based
diffing tools, but may break the logical sequence of nodes in
`/tmp/exec_graph2.log.txt`.

For large log files, you might need to increase the JVM heap size. To do so,
pass a corresponding `--jvm_flag`, for example `--jvm_flag=-Xmx4g` for 4GB of heap.
