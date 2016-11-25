---
layout: posts
title: Using Bazel in a continuous integration system
---

When doing continuous integration, you do not want your build to fail because a
a tool invoked during the build has been updated or some environmental
conditions have changed. Because Bazel is designed for reproducible builds and
keeps track of almost every dependency of your project, Bazel is a great tool
for use inside a CI system. Bazel also caches results of previous build,
including test results and will not re-run unchanged tests, speeding up each
build.

## Running Bazel on virtual or physical machines.

For [ci.bazel.build](http://ci.bazel.build), we use
[Google Compute Engine](https://cloud.google.com/compute/) virtual machine for
our Linux build and a physical Mac mini for our Mac build. Apart from Bazel
tests that are run using the
[`./compile.sh`](https://github.com/bazelbuild/bazel/blob/master/compile.sh)
script, we also run some projects to validate Bazel binaries against: the
[Bazel Tutorial](https://github.com/bazelbuild/examples/tree/master/tutorial)
[here](http://ci.bazel.build/job/Tutorial/),
[re2](https://github.com/google/re2) [here](http://ci.bazel.build/job/re2/),
[protobuf](https://github.com/google/protobuf)
[here](http://ci.bazel.build/job/protobuf/), and
[TensorFlow](https://www.tensorflow.org)
[here](http://ci.bazel.build/job/TensorFlow/).

Bazel is reinstalled each time we run the tutorial or TensorFlow, but the Bazel
cache is maintained across installs. The setup for those jobs is the following:

```bash
set -e

# Fetch the Bazel installer
URL=https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-${INSTALLER_PLATFORM}.sh
export BAZEL_INSTALLER=${PWD}/bazel-installer/install.sh
curl -L -o ${BAZEL_INSTALLER} ${URL}
BASE="${PWD}/bazel-install"

# Install bazel inside ${BASE}
bash "${BAZEL_INSTALLER}" \
  --base="${BASE}" \
  --bazelrc="${BASE}/bin/bazel.bazelrc" \
  --bin="${BASE}/binary"

# Run the build
BAZEL="${BASE}/binary/bazel --bazelrc=${BASE}/bin/bazel.bazelrc"
${BAZEL} test //...
```

This tests installing a specific version of Bazel each time. Of course, if
Bazel is installed on the path, one can simply `bazel test //...`. However,
even with reinstalling all the time, Bazel caching simply works.


## Running Bazel inside a Docker container

Several people want to use Bazel in a Docker container. First of all, Bazel
has some feature that are incompatibles with Docker:

- Bazel runs by default in client/server mode using UNIX domain sockets, so if
  you cannot mount the socket inside the Docker container, then you must disable
  client-server communication by running Bazel in batch mode with the `--batch`
  flag.
- Bazel [sandboxes all actions on linux by default](http://bazel.build/blog/2015/09/11/sandboxing.html)
  and this needs special privileges in the Docker container (enabled by
  [`--privilege=true`](https://docs.docker.com/engine/reference/run/#runtime-privilege-linux-capabilities-and-lxc-configuration).
  If you cannot enable the namespace sandbox, you can deactivate it in Bazel
  with the `--genrule_strategy=standalone --spawn_strategy=standalone` flags.

So the last step of the previous script would look like:

```bash
# Run the build
BAZEL="${BASE}/binary/bazel --bazelrc=${BASE}/bin/bazel.bazelrc --batch"
${BAZEL} test --genrule_strategy=standalone --spawn_strategy=standalone \
    //...
```

This build will however be slower because the server has to restart for every
build and the cache will be lost when the Docker container is destroyed.

To prevent the loss of the cache, it is better to mount a persistent volume for
`~/.cache/bazel` (where the Bazel cache is stored).


## Return code and XML output

A final consideration when setting up a continuous integration system is getting
the result from the build. Bazel has the following interesting exit codes when
using `test` and `build` commands:

<table class="table table-condensed table-striped">
  <thead>
    <tr>
      <th>Exit Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Success.</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Build failed.</td>
    </tr>
    <tr>
      <td>2</td>
      <td>
        Command Line Problem, Bad or Illegal flags or command combination, or
        Bad Environment Variables. Your command line must be modified.
      </td>
    </tr>
    <tr>
      <td>3</td>
      <td>Build OK, but some tests failed or timed out.</td>
    </tr>
    <tr>
      <td>4</td>
      <td>
        Build successful but no tests were found even though testing was
        requested.
      </td>
    </tr>
    <tr>
      <td>8</td>
      <td>
        Build interrupted (by a <kbd><kbd>Ctrl</kbd>+<kbd>C</kbd></kbd> from the
        user for instance) but we terminated with an orderly shutdown.
      </td>
    </tr>
  </tbody>
</table>

These return codes can be used to determine the reason for a failure
(in [ci.bazel.build](http://ci.bazel.build), we mark builds that have exited with exit
code 3 as unstable, and other non zero code as failed).

You can also control how much information about test results Bazel prints out
with the [`--test_output` flag](http://bazel.build/docs/bazel-user-manual.html#flag--test_output).
Generally, printing the output of test that fails with `--test_output=errors` is
a good setting for a CI system.

Finally, Bazel's built-in [JUnit test runner](https://github.com/bazelbuild/bazel/blob/master/src/java_tools/junitrunner)
generates Ant-style XML output file (in `bazel-testlogs/pkg/target/test.xml`)
that summarizes the results of your tests. This test runner can be activated
with the `--nolegacy_bazel_java_test` flag (this will soon be the default).
Other tests also get [a basic XML output file](https://github.com/bazelbuild/bazel/blob/master/tools/test/test-setup.sh#L54)
that contains only the result of the test (success or failure).

To get your test results, you can also use the
[Bazel dashboard](http://bazel.build/blog/2015/07/29/dashboard-dogfood.html),
an optional system that automatically uploads Bazel test results to a shared
server.
