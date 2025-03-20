Project: /_project.yaml
Book: /_book.yaml

# Calling Bazel from scripts

{% include "_buttons.html" %}

You can call Bazel from scripts to perform a build, run tests, or query
the dependency graph. Bazel has been designed to enable effective scripting, but
this section lists some details to bear in mind to make your scripts more
robust.

### Choosing the output base {:#output-base-option}

The `--output_base` option controls where the Bazel process should write the
outputs of a build to, as well as various working files used internally by
Bazel, one of which is a lock that guards against concurrent mutation of the
output base by multiple Bazel processes.

Choosing the correct output base directory for your script depends on several
factors. If you need to put the build outputs in a specific location, this will
dictate the output base you need to use. If you are making a "read only" call to
Bazel (such as `bazel query`), the locking factors will be more important. In
particular, if you need to run multiple instances of your script concurrently,
you should be mindful that each Blaze server process can handle at most one
invocation [at a time](/run/client-server#clientserver-implementation).
Depending on your situation it may make sense for each instance of your script
to wait its turn, or it may make sense to use `--output_base` to run multiple
Blaze servers and use those.

If you use the default output base value, you will be contending for the same
lock used by the user's interactive Bazel commands. If the user issues
long-running commands such as builds, your script will have to wait for those
commands to complete before it can continue.

### Notes about server mode {:#server-mode}

By default, Bazel uses a long-running [server process](/run/client-server) as an
optimization. When running Bazel in a script, don't forget to call `shutdown`
when you're finished with the server, or, specify `--max_idle_secs=5` so that
idle servers shut themselves down promptly.

### What exit code will I get? {:#exit-codes}

Bazel attempts to differentiate failures due to the source code under
consideration from external errors that prevent Bazel from executing properly.
Bazel execution can result in following exit codes:

**Exit Codes common to all commands:**

-   `0` - Success
-   `2` - Command Line Problem, Bad or Illegal flags or command combination, or
    Bad Environment Variables. Your command line must be modified.
-   `8` - Build Interrupted but we terminated with an orderly shutdown.
-   `9` - The server lock is held and `--noblock_for_lock` was passed.
-   `32` - External Environment Failure not on this machine.

-   `33` - Bazel ran out of memory and crashed. You need to modify your command line.
-   `34` - Reserved for Google-internal use.
-   `35` - Reserved for Google-internal use.
-   `36` - Local Environmental Issue, suspected permanent.
-   `37` - Unhandled Exception / Internal Bazel Error.
-   `38` - Transient error publishing results to the Build Event Service.
-   `39` - Blobs required by Bazel are evicted from Remote Cache.
-   `41-44` - Reserved for Google-internal use.
-   `45` - Persistent error publishing results to the Build Event Service.
-   `47` - Reserved for Google-internal use.
-   `49` - Reserved for Google-internal use.

**Return codes for commands `bazel build`, `bazel test`:**

-   `1` - Build failed.
-   `3` - Build OK, but some tests failed or timed out.
-   `4` - Build successful but no tests were found even though testing was
    requested.


**For `bazel run`:**

-   `1` - Build failed.
-   If the build succeeds but the executed subprocess returns a non-zero exit
    code it will be the exit code of the command as well.

**For `bazel query`:**

-   `3` - Partial success, but the query encountered 1 or more errors in the
    input BUILD file set and therefore the results of the operation are not 100%
    reliable. This is likely due to a `--keep_going` option on the command line.
-   `7` - Command failure.

Future Bazel versions may add additional exit codes, replacing generic failure
exit code `1` with a different non-zero value with a particular meaning.
However, all non-zero exit values will always constitute an error.


### Reading the .bazelrc file {:#reading-bazelrc}

By default, Bazel reads the [`.bazelrc` file](/run/bazelrc) from the base
workspace directory or the user's home directory. Whether or not this is
desirable is a choice for your script; if your script needs to be perfectly
hermetic (such as when doing release builds), you should disable reading the
.bazelrc file by using the option `--bazelrc=/dev/null`. If you want to perform
a build using the user's preferred settings, the default behavior is better.

### Command log {:#command-log}

The Bazel output is also available in a command log file which you can find with
the following command:

```posix-terminal
bazel info command_log
```

The command log file contains the interleaved stdout and stderr streams of the
most recent Bazel command. Note that running `bazel info` will overwrite the
contents of this file, since it then becomes the most recent Bazel command.
However, the location of the command log file will not change unless you change
the setting of the `--output_base` or `--output_user_root` options.

### Parsing output {:#parsing-output}

The Bazel output is quite easy to parse for many purposes. Two options that may
be helpful for your script are `--noshow_progress` which suppresses progress
messages, and <code>--show_result <var>n</var></code>, which controls whether or
not "build up-to-date" messages are printed; these messages may be parsed to
discover which targets were successfully built, and the location of the output
files they created. Be sure to specify a very large value of _n_ if you rely on
these messages.

## Troubleshooting performance by profiling {:#performance-profiling}

See the [Performance Profiling](/rules/performance#performance-profiling) section.
