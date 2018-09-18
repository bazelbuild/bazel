# Workspace Log Parser

This tool is used to inspect and parse the Bazel workspace logs.
To generate the workspace log, run e.g.,:

        bazel build \
            --experimental_workspace_rules_log_file=/tmp/workspace.log :hello_world

Then build the parser and run it.

        bazel build src/tools/workspacelog:all
        bazel-bin/src/tools/workspacelog/parser --log_path=/tmp/workspace.log

This will simply print the log contents to stdout in text form.


To output results to a file, use `--output_path`:

        bazel-bin/src/tools/workspacelog/parser --log_path=/tmp/workspace.log \
            --output_path=/tmp/workspace.log.txt


To exclude all events produced by a certain rule, use `--exclude_rule`:

        bazel build src/tools/workspacelog:all
        bazel-bin/src/tools/workspacelog/parser --log_path=/tmp/workspace.log \
            --exclude_rule "//external:local_config_cc"

Note that `--exclude_rule` may be specified multiple times.

        bazel build src/tools/workspacelog:all
        bazel-bin/src/tools/workspacelog/parser --log_path=/tmp/workspace.log \
            --exclude_rule "//external:local_config_cc" \
            --exclude_rule "//external:dep"

For example, the above will filter out any events produced by rules
`//external:local_config_cc` or `//external:dep`

