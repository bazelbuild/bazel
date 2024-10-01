Project: /_project.yaml
Book: /_book.yaml

# Debugging Remote Cache Hits for Remote Execution

{% include "_buttons.html" %}

This page describes how to check your cache hit rate and how to investigate
cache misses in the context of remote execution.

This page assumes that you have a build and/or test that successfully
utilizes remote execution, and you want to ensure that you are effectively
utilizing remote cache.

## Checking your cache hit rate {:#check-cache-hits}

In the standard output of your Bazel run, look at the `INFO` line that lists
processes, which roughly correspond to Bazel actions. That line details
where the action was run. Look for the `remote` label, which indicates an action
executed remotely, `linux-sandbox` for actions executed in a local sandbox,
and other values for other execution strategies. An action whose result came
from a remote cache is displayed as `remote cache hit`.

For example:

```none {:.devsite-disable-click-to-copy}
INFO: 11 processes: 6 remote cache hit, 3 internal, 2 remote.
```

In this example there were 6 remote cache hits, and 2 actions did not have
cache hits and were executed remotely. The 3 internal part can be ignored.
It is typically tiny internal actions, such as creating symbolic links. Local
cache hits are not included in this summary. If you are getting 0 processes
(or a number lower than expected), run `bazel clean` followed by your build/test
command.

## Troubleshooting cache hits {:#troubleshooting-cache-hits}

If you are not getting the cache hit rate you are expecting, do the following:

### Ensure re-running the same build/test command produces cache hits {:#rerun-cache-hits}

1. Run the build(s) and/or test(s) that you expect to populate the cache. The
   first time a new build is run on a particular stack, you can expect no remote
   cache hits. As part of remote execution, action results are stored in the
   cache and a subsequent run should pick them up.

2. Run `bazel clean`. This command cleans your local cache, which allows
   you to investigate remote cache hits without the results being masked by
   local cache hits.

3. Run the build(s) and test(s) that you are investigating again (on the same
   machine).

4. Check the `INFO` line for cache hit rate. If you see no processes except
   `remote cache hit` and `internal`, then your cache is being correctly populated and
   accessed. In that case, skip to the next section.

5. A likely source of discrepancy is something non-hermetic in the build causing
   the actions to receive different action keys across the two runs. To find
   those actions, do the following:

   a. Re-run the build(s) or test(s) in question to obtain execution logs:

      ```posix-terminal
      bazel clean

      bazel {{ '<var>' }}--optional-flags{{ '</var>' }} build //{{ '<var>' }}your:target{{ '</var>' }} --execution_log_compact_file=/tmp/exec1.log
      ```

   b. [Compare the execution logs](#compare-logs-the-execution-logs) between the
      two runs. Ensure that the actions are identical across the two log files.
      Discrepancies provide a clue about the changes that occurred between the
      runs. Update your build to eliminate those discrepancies.

   If you are able to resolve the caching problems and now the repeated run
   produces all cache hits, skip to the next section.

   If your action IDs are identical but there are no cache hits, then something
   in your configuration is preventing caching. Continue with this section to
   check for common problems.

5. Check that all actions in the execution log have `cacheable` set to true. If
   `cacheable` does not appear in the execution log for a give action, that
   means that the corresponding rule may have a `no-cache` tag in its
   definition in the `BUILD` file. Look at the `mnemonic` and `target_label`
   fields in the execution log to help determine where the action is coming
   from.

6. If the actions are identical and `cacheable` but there are no cache hits, it
   is possible that your command line includes `--noremote_accept_cached` which
   would disable cache lookups for a build.

   If figuring out the actual command line is difficult, use the canonical
   command line from the
   [Build Event Protocol](/remote/bep)
   as follows:

   a. Add `--build_event_text_file=/tmp/bep.txt` to your Bazel command to get
    the text version of the log.

   b. Open the text version of the log and search for the
    `structured_command_line` message with `command_line_label: "canonical"`.
    It will list all the options after expansion.

   c. Search for `remote_accept_cached` and check whether it's set to `false`.

   d. If `remote_accept_cached` is `false`, determine where it is being
      set to `false`: either at the command line or in a
      [bazelrc](/run/bazelrc#bazelrc-file-locations) file.

### Ensure caching across machines {:#caching-across-machines}

After cache hits are happening as expected on the same machine, run the
same build(s)/test(s) on a different machine. If you suspect that caching is
not happening across machines, do the following:

1. Make a small modification to your build to avoid hitting existing caches.

2. Run the build on the first machine:

   ```posix-terminal
    bazel clean

    bazel ... build ... --execution_log_compact_file=/tmp/exec1.log
   ```

3. Run the build on the second machine, ensuring the modification from step 1
   is included:

   ```posix-terminal
    bazel clean

    bazel ... build ... --execution_log_compact_file=/tmp/exec2.log
   ```

4. [Compare the execution logs](#compare-logs-the-execution-logs) for the two
    runs. If the logs are not identical, investigate your build configurations
    for discrepancies as well as properties from the host environment leaking
    into either of the builds.

## Comparing the execution logs {:#compare-logs}

The execution log contains records of actions executed during the build.
Each record describes both the inputs (not only files, but also command line
arguments, environment variables, etc) and the outputs of the action. Thus,
examination of the log can reveal why an action was reexecuted.

The execution log can be produced in one of three formats:
compact (`--execution_log_compact_file`),
binary (`--execution_log_binary_file`) or JSON (`--execution_log_json_file`).
The compact format is recommended, as it produces much smaller files with very
little runtime overhead. The following instructions work for any format. You
can also convert between them using the `//src/tools/execlog:converter` tool.

To compare logs for two builds that are not sharing cache hits as expected,
do the following:

1. Get the execution logs from each build and store them as `/tmp/exec1.log` and
   `/tmp/exec2.log`.

2. Download the Bazel source code and build the `//src/tools/execlog:parser`
   tool:

       git clone https://github.com/bazelbuild/bazel.git
       cd bazel
       bazel build //src/tools/execlog:parser

3. Use the `//src/tools/execlog:parser` tool to convert the logs into a
   human-readable text format. In this format, the actions in the second log are
   sorted to match the order in the first log, making a comparison easier.

        bazel-bin/src/tools/execlog/parser \
          --log_path=/tmp/exec1.log \
          --log_path=/tmp/exec2.log \
          --output_path=/tmp/exec1.log.txt \
          --output_path=/tmp/exec2.log.txt

4. Use your favourite text differ to diff `/tmp/exec1.log.txt` and
   `/tmp/exec2.log.txt`.
