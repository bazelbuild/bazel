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

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
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
  public void createDispatcher() throws Exception {
    BlazeRuntime runtime = getRuntime();
    runtime.getCommandMap().put("dump", new DumpCommand());
    dispatcher = new BlazeCommandDispatcher(runtime);
    write("foo/BUILD", "genrule(name = 'foo', outs = ['out'], cmd = 'touch $@')");
    addOptions("--nobuild");
    buildTarget("//foo:foo");
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
  public void skyframeChanged() throws Exception {
    // The target was already built once, so only the changes of the second build are dumped.
    write("foo/BUILD", "genrule(name = 'foo', outs = ['out'], cmd = 'touch $@', tags = ['a'])");
    buildTarget("//foo:foo");

    assertThat(dump("--skyframe", "changed").isSuccess()).isTrue();
    var out = recordingOutErr.outAsLatin1();
    assertThat(out).containsMatch("(?m)^ +[0-9]+ PACKAGE$");
    assertThat(out).containsMatch("(?m)^ +[0-9]+ CONFIGURED_TARGET$");
    // The modified BUILD file is an origin of the changes.
    assertThat(out).containsMatch("(?m)^FILE_STATE:\\[.*\\]/\\[foo/BUILD\\]$");
    // The package changed because its BUILD file did, and the target because its package did.
    assertThat(out).containsMatch("(?m)^PACKAGE:foo\n    FILE:\\[.*\\]/\\[foo/BUILD\\]\n");
    assertThat(out).containsMatch("(?m)^CONFIGURED_TARGET:.*//foo:foo.*\n(    .*\n)*    PACKAGE:foo\n");
    // The directory of the package wasn't touched by either build.
    assertThat(out).doesNotContain("]/[foo]\n");

    // The filter restricts the origins and the detailed list, but not the counts.
    recordingOutErr = new RecordingOutErr();
    assertThat(dump("--skyframe", "changed", "--skykey_filter", "^PACKAGE:").isSuccess()).isTrue();
    out = recordingOutErr.outAsLatin1();
    assertThat(out).containsMatch("(?m)^ +[0-9]+ CONFIGURED_TARGET$");
    assertThat(out).contains("\nPACKAGE:foo\n    FILE:");
    assertThat(out).doesNotContain("\nFILE_STATE:");
    assertThat(out).doesNotContain("\nCONFIGURED_TARGET:");
  }

  @Test
  public void multiOptionSmoke() throws Exception {
    assertThat(dump("--rule_classes", "--rules", "--skyframe", "summary").isSuccess()).isTrue();
    assertThat(recordingOutErr.outAsLatin1()).contains("filegroup");
    assertThat(recordingOutErr.outAsLatin1()).contains("RULE");
    assertThat(recordingOutErr.outAsLatin1()).contains("Node count");
  }
}
