// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import java.util.Collections;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DumpCommand}. */
@RunWith(JUnit4.class)
public final class DumpCommandTest extends BuildIntegrationTestCase {
  private BlazeCommandDispatcher dispatcher;
  private RecordingOutErr recordingOutErr;

  @Before
  public void createDispatcher() {
    BlazeRuntime runtime = getRuntime();
    runtime.getCommandMap().put("dump", new DumpCommand());
    dispatcher = new BlazeCommandDispatcher(runtime);
  }

  @Before
  public void createRecording() throws Exception {
    recordingOutErr = new RecordingOutErr();
  }

  private BlazeCommandResult dump(String... args) throws InterruptedException {
    List<String> params = Lists.newArrayList("dump");
    Collections.addAll(params, args);
    return dispatcher.exec(params, "test", recordingOutErr);
  }

  @Test
  public void doesNotContainWarningInStdout() throws Exception {
    assertThat(dump("--skyframe", "count").isSuccess()).isTrue();
    assertThat(recordingOutErr.errAsLatin1()).contains(DumpCommand.WARNING_MESSAGE);
    assertThat(recordingOutErr.outAsLatin1()).doesNotContain(DumpCommand.WARNING_MESSAGE);
  }

  @Test
  public void multiOptionSmoke() throws Exception {
    write("foo/BUILD", "genrule(name = 'foo', outs = ['out'], cmd = 'touch $@')");
    addOptions("--nobuild");
    buildTarget("//foo:foo");
    assertThat(dump("--rule_classes", "--rules", "--skyframe", "summary").isSuccess()).isTrue();
    assertThat(recordingOutErr.outAsLatin1()).contains("filegroup");
    assertThat(recordingOutErr.outAsLatin1()).contains("RULE");
    assertThat(recordingOutErr.outAsLatin1()).contains("Node count");
  }

  @Test
  public void interruptedOnRuleTypes() throws Exception {
    write(
        "foo/BUILD",
        "genrule(name = 'bar', outs = ['bar.out'], cmd = 'touch $@')",
        "genrule(name = 'bad', outs = ['bad.out'], srcs = [':bar'], cmd = '$BAD_VARIABLE')",
        "genrule(name = 'foo', outs = ['foo.out'], srcs = [':bad'], cmd = 'touch $@')");
    addOptions("--nobuild");
    // The analysis failure enqueues nodes for invalidation/deletion, which get triggered during the
    // getPackage() Skyframe evaluations during the rule-type dump. The interrupt then kicks in.
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:foo"));
    Thread.currentThread().interrupt();
    BlazeCommandResult result = dump("--rule_classes", "--rules", "--skyframe", "summary");
    assertThat(result.getExitCode()).isEqualTo(ExitCode.INTERRUPTED);
    assertThat(recordingOutErr.errAsLatin1()).contains("Interrupted");
    // Dump output that happened before the interrupt is still there: this is sensitive to the order
    // DumpCommand processes its options in.
    assertThat(recordingOutErr.outAsLatin1()).contains("filegroup");
    // There is no rule type output.
    assertThat(recordingOutErr.outAsLatin1()).doesNotContain("RULE");
    // Later dump output does not happen.
    assertThat(recordingOutErr.outAsLatin1()).doesNotContain("Node count");
  }
}
