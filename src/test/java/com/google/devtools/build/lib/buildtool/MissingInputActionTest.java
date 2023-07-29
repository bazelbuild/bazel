// Copyright 2020 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.bazel.BazelWorkspaceStatusModule;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.vfs.Path;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests related to "missing input file" errors. */
@RunWith(JUnit4.class)
public final class MissingInputActionTest extends BuildIntegrationTestCase {

  @Override
  protected BlazeModule getBuildInfoModule() {
    return new BazelWorkspaceStatusModule();
  }

  // Regression test for bug #904676: Blaze does not consider missing inputs
  // an error.
  @Test
  public void testNoInput() throws Exception {
    // Multiple missing inputs means error is non-deterministic in --nokeep_going case.
    this.addOptions("--keep_going");
    write("dummy/BUILD",
          "genrule(name = 'dummy', ",
          "        srcs = ['in1', 'in2', 'in3'], ",
          "        outs = ['out1', 'out2'],  ",
          "        cmd = '/bin/true')");
    write("dummy/in1");

    assertMissingInputOnBuild("//dummy", 2);
    events.assertDoesNotContainEvent("missing input file '" + "//" + "dummy" + ":" + "in1'");
    events.assertContainsError("missing input file '" + "//" + "dummy" + ":" + "in2'");
    events.assertContainsError("missing input file '" + "//" + "dummy" + ":" + "in3'");
  }

  // The next two tests are inherently flakily successful with respect to the workspace status
  // action: even if we don't correctly suppress the workspace status action error message, we might
  // not have started it at all because Skyframe aborted quickly. That doesn't happen in practice,
  // though: the workspace status action starts right away.

  @Test
  public void testMissingInputRacesWithWorkspaceStatusAction() throws Exception {
    write(
        "dummy/BUILD",
        "genrule(name = 'dummy', srcs = ['in'], outs = ['out'], cmd = '/bin/false')");
    Path sleepPath = write("sleep.sh", "sleep 300");
    sleepPath.setExecutable(true);
    addOptions("--workspace_status_command=" + sleepPath.getPathString());
    for (int i = 0; i < 2; i++) {
      assertMissingInputOnBuild("//dummy", 1);
      events.assertContainsError(
          "dummy/BUILD:1:8: Executing genrule //dummy:dummy failed: missing input file"
              + " '//dummy:in'");
      events.assertContainsEventWithFrequency("missing input file", 1);
      events.assertDoesNotContainEvent("Failed to determine build info");
      events.clear();
    }
  }

  @Test
  public void testMissingTopLevelInputRacesWithWorkspaceStatusAction() throws Exception {
    // Create a rule that exports a missing source file as a top-level artifact so that the missing
    // file will be detected by the TargetCompletion function, not an ActionExecution function.
    write(
        "foo/missing.bzl",
        "def _missing_impl(ctx):",
        "    return DefaultInfo(files = depset(ctx.files.srcs))",
        "",
        "missing = rule(",
        "               implementation = _missing_impl,",
        "               attrs = { 'srcs': attr.label_list(allow_files = True) }",
        ")");
    write(
        "foo/BUILD",
        "load('missing.bzl', 'missing')",
        "missing(name = 'foo', srcs = ['missing.sh'])");
    Path sleepPath = write("sleep.sh", "sleep 300");
    sleepPath.setExecutable(true);
    addOptions("--workspace_status_command=" + sleepPath.getPathString());
    for (int i = 0; i < 2; i++) {
      assertMissingInputOnBuild("//foo:foo", 1);
      events.assertContainsError("foo/BUILD:2:8: //foo:foo: missing input file '//foo:missing.sh'");
      events.assertContainsEventWithFrequency("missing input file", 1);
      events.assertDoesNotContainEvent("Failed to determine build info");
      events.clear();
    }
  }

  @Test
  public void testMissingTopLevelInput() throws Exception {
    // Create a rule that exports a missing source file as a top-level artifact so that the missing
    // file will be detected by the TargetCompletion function, not an ActionExecution function.
    write(
        "foo/missing.bzl",
        "def _missing_impl(ctx):",
        "    return DefaultInfo(files = depset(ctx.files.srcs))",
        "",
        "missing = rule(",
        "               implementation = _missing_impl,",
        "               attrs = { 'srcs': attr.label_list(allow_files = True) }",
        ")");
    write(
        "foo/BUILD",
        "load('missing.bzl', 'missing')",
        "missing(name = 'foo', srcs = ['missing.sh'])");
    addOptions("--keep_going");
    assertMissingInputOnBuild("//foo:foo", 1);
    events.assertContainsError("foo/BUILD:2:8: //foo:foo: missing input file '//foo:missing.sh'");
    events.assertContainsEventWithFrequency("missing input file", 1);
  }

  private void assertMissingInputOnBuild(String label, int numMissing) {
    BuildFailedException e = assertThrows(BuildFailedException.class, () -> buildTarget(label));
    FailureDetail failureDetail = e.getDetailedExitCode().getFailureDetail();
    assertThat(failureDetail.getExecution().getCode()).isEqualTo(Code.SOURCE_INPUT_MISSING);
    String expected = numMissing + " input file(s) do not exist";
    events.assertContainsError(expected);
    events.assertContainsEventWithFrequency(expected, 1);
    events.assertContainsError(label);
  }

  @Test
  public void allErrorsAggregated() throws Exception {
    write(
        "foo/BUILD",
        "genrule(name = 'foo', srcs = [':in', ':genin'], outs = ['out'], cmd = 'touch $@')",
        "genrule(name = 'gen', outs = ['genin'], cmd = 'false')");
    AtomicReference<TargetCompleteEvent> targetCompleteEventRef = new AtomicReference<>();
    runtimeWrapper.registerSubscriber(
        new Object() {
          @SuppressWarnings("unused")
          @Subscribe
          public void accept(TargetCompleteEvent event) {
            targetCompleteEventRef.set(event);
          }
        });
    RecordingOutErr outErr = new RecordingOutErr();
    this.outErr = outErr;
    addOptions("--keep_going");
    assertThrows(BuildFailedException.class, () -> buildTarget("//foo:foo"));
    assertThat(targetCompleteEventRef.get().getRootCauses().toList()).hasSize(1);
    assertThat(outErr.errAsLatin1()).contains("Executing genrule //foo:gen failed");
  }
}
