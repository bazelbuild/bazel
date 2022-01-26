// Copyright 2021 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for Starlark "unused inputs list" functionality on failures caused by unused inputs. */
@RunWith(TestParameterInjector.class)
public final class UnusedInputsFailureIntegrationTest extends BuildIntegrationTestCase {

  @TestParameter private boolean keepGoing;

  @Before
  public void setOptions() {
    addOptions("--keep_going=" + keepGoing);
  }

  @Test
  public void incrementalFailureOnUnusedInput() throws Exception {
    RecordingBugReporter bugReporter = recordBugReportsAndReinitialize();
    write(
        "foo/pruning.bzl",
        "def _impl(ctx):",
        "  inputs = ctx.attr.inputs.files",
        "  output = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  unused_file = ctx.actions.declare_file(ctx.label.name + '.unused')",
        "  ctx.actions.run(",
        "    # Make sure original inputs are one level down, so 'leaf unrolling' doesn't get them",
        "    inputs = depset(transitive = [ctx.attr.filler.files, inputs]),",
        "    outputs = [output, unused_file],",
        "    arguments = [output.path, unused_file.path] + [f.path for f in inputs.to_list()],",
        "    executable = ctx.executable.executable,",
        "    unused_inputs_list = unused_file,",
        "  )",
        "  return DefaultInfo(files = depset([output]))",
        "",
        "build_rule = rule(",
        "  attrs = {",
        "    'inputs': attr.label(allow_files = True),",
        "    'filler': attr.label(allow_files = True),",
        "    'executable': attr.label(executable = True, allow_files = True, cfg = 'host'),",
        "  },",
        "  implementation = _impl,",
        ")");
    write("foo/unused.sh", "touch $1", "shift", "unused=$1", "shift", "echo $@ > $unused")
        .setExecutable(true);
    write("foo/gen_run.sh", "true").setExecutable(true);
    write("foo/filler");
    write(
        "foo/BUILD",
        "load('//foo:pruning.bzl', 'build_rule')",
        "build_rule(name = 'foo', filler = ':filler', inputs = ':in', executable = ':unused.sh')",
        "genrule(name = 'gen', outs = ['in'], tools = [':gen_run.sh'], cmd = '$(location"
            + " :gen_run.sh) && touch $@')");
    buildTarget("//foo:foo");
    bugReporter.assertNoExceptions();

    write("foo/gen_run.sh", "false");
    if (keepGoing) {
      buildTarget("//foo:foo");
      bugReporter.assertNoExceptions();
    } else {
      RecordingOutErr outErr = new RecordingOutErr();
      this.outErr = outErr;
      BuildFailedException e = assertThrows(BuildFailedException.class, () -> buildTarget("//foo"));
      assertThat(e.getDetailedExitCode().getFailureDetail())
          .comparingExpectedFieldsOnly()
          .isEqualTo(
              FailureDetails.FailureDetail.newBuilder()
                  .setExecution(
                      FailureDetails.Execution.newBuilder()
                          .setCode(FailureDetails.Execution.Code.UNEXPECTED_EXCEPTION)
                          .build())
                  .build());
      assertThat(outErr.errAsLatin1()).contains("Executing genrule //foo:gen failed");
      Throwable cause = bugReporter.getFirstCause();
      assertThat(cause).hasMessageThat().contains("Error evaluating artifact nested set");
      assertThat(cause).hasMessageThat().contains("foo/gen_run.sh");
    }
  }

  @Test
  public void incrementalUnusedSymlinkCycle() throws Exception {
    RecordingBugReporter bugReporter = recordBugReportsAndReinitialize();
    write(
        "foo/pruning.bzl",
        "def _impl(ctx):",
        "  inputs = ctx.attr.inputs.files",
        "  output = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  unused_inputs_list = ctx.actions.declare_file(ctx.label.name + '.unused')",
        "  arguments = [output.path, unused_inputs_list.path]",
        "  for input in inputs.to_list():",
        "    arguments += [input.path]",
        "  ctx.actions.run(",
        "    inputs = inputs,",
        "    outputs = [output, unused_inputs_list],",
        "    arguments = arguments,",
        "    executable = ctx.executable.executable,",
        "    unused_inputs_list = unused_inputs_list,",
        "  )",
        "  return DefaultInfo(files = depset([output]))",
        "",
        "build_rule = rule(",
        "  attrs = {",
        "    'inputs': attr.label(allow_files = True),",
        "    'executable': attr.label(executable = True, allow_files = True, cfg = 'host'),",
        "  },",
        "  implementation = _impl,",
        ")");
    Path unusedSh =
        write("foo/all_unused.sh", "touch $1", "shift", "unused=$1", "shift", "echo $@ > $unused");
    unusedSh.setExecutable(true);
    Path inPath = write("foo/in");
    write(
        "foo/BUILD",
        "load('//foo:pruning.bzl', 'build_rule')",
        "build_rule(name = 'prune', inputs = ':in', executable = ':all_unused.sh')");
    buildTarget("//foo:prune");
    bugReporter.assertNoExceptions();

    inPath.delete();
    inPath.createSymbolicLink(PathFragment.create("in"));
    if (keepGoing) {
      buildTarget("//foo:prune");
      bugReporter.assertNoExceptions();
    } else {
      RecordingOutErr outErr = new RecordingOutErr();
      this.outErr = outErr;
      BuildFailedException e =
          assertThrows(BuildFailedException.class, () -> buildTarget("//foo:prune"));
      assertDetailedExitCodeIsSourceIOFailure(e);
      Throwable cause = bugReporter.getFirstCause();
      assertDetailedExitCodeIsSourceIOFailure(cause);
      assertThat(cause).hasMessageThat().isEqualTo("error reading file '//foo:in': Symlink cycle");
      assertThat(outErr.errAsLatin1()).contains("error reading file '//foo:in': Symlink cycle");
    }
  }

  private static final FailureDetails.FailureDetail SOURCE_IO_FAILURE =
      FailureDetails.FailureDetail.newBuilder()
          .setExecution(
              FailureDetails.Execution.newBuilder()
                  .setCode(FailureDetails.Execution.Code.SOURCE_INPUT_IO_EXCEPTION))
          .build();

  private static void assertDetailedExitCodeIsSourceIOFailure(Throwable exception) {
    assertThat(exception).isInstanceOf(DetailedException.class);
    assertThat(((DetailedException) exception).getDetailedExitCode().getFailureDetail())
        .comparingExpectedFieldsOnly()
        .isEqualTo(SOURCE_IO_FAILURE);
  }

  @Test
  public void incrementalUnusedDanglingSymlink() throws Exception {
    write(
        "foo/pruning.bzl",
        "def _impl(ctx):",
        "  inputs = ctx.attr.inputs.files",
        "  output = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  unused_inputs_list = ctx.actions.declare_file(ctx.label.name + '.unused')",
        "  arguments = [output.path, unused_inputs_list.path]",
        "  for input in inputs.to_list():",
        "    arguments += [input.path]",
        "  ctx.actions.run(",
        "    inputs = inputs,",
        "    outputs = [output, unused_inputs_list],",
        "    arguments = arguments,",
        "    executable = ctx.executable.executable,",
        "    unused_inputs_list = unused_inputs_list,",
        "  )",
        "  return DefaultInfo(files = depset([output]))",
        "",
        "build_rule = rule(",
        "  attrs = {",
        "    'inputs': attr.label(allow_files = True),",
        "    'executable': attr.label(executable = True, allow_files = True, cfg = 'host'),",
        "  },",
        "  implementation = _impl,",
        ")");
    Path unusedSh =
        write("foo/all_unused.sh", "touch $1", "shift", "unused=$1", "shift", "echo $@ > $unused");
    unusedSh.setExecutable(true);
    Path inPath = write("foo/in");
    write(
        "foo/BUILD",
        "load('//foo:pruning.bzl', 'build_rule')",
        "build_rule(name = 'prune', inputs = ':in', executable = ':all_unused.sh')");
    buildTarget("//foo:prune");
    inPath.delete();
    inPath.createSymbolicLink(PathFragment.create("nope"));
    buildTarget("//foo:prune");
  }
}
