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
package com.google.devtools.build.lib.remote;

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.remote.util.IntegrationTestUtils.startWorker;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.readContent;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils.WorkerInstance;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BuildSummaryStatsModule;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

/** Integration tests for Build without the Bytes. */
@RunWith(Parameterized.class)
public class BuildWithoutTheBytesIntegrationTest extends BuildIntegrationTestCase {
  public enum RemoteMode {
    REMOTE_EXECUTION,
    REMOTE_CACHE;

    public boolean executeRemotely() {
      return this == REMOTE_EXECUTION;
    }
  }

  public enum OutputMode {
    DOWNLOAD_TOPLEVEL,
    DOWNLOAD_MINIMAL;

    public boolean minimal() {
      return this == DOWNLOAD_MINIMAL;
    }
  }

  private WorkerInstance worker;
  private final RemoteMode remoteMode;
  private final OutputMode outputMode;

  @Parameterized.Parameters(name = "{0}-{1}")
  public static List<Object[]> configs() {
    ArrayList<Object[]> params = new ArrayList<>();
    for (RemoteMode remoteMode : RemoteMode.values()) {
      for (OutputMode outputMode : OutputMode.values()) {
        params.add(new Object[] {remoteMode, outputMode});
      }
    }
    return params;
  }

  public BuildWithoutTheBytesIntegrationTest(RemoteMode remoteMode, OutputMode outputMode) {
    this.remoteMode = remoteMode;
    this.outputMode = outputMode;
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder().addBlazeModule(new BuildSummaryStatsModule());
  }

  @Override
  protected ImmutableList<BlazeModule> getSpawnModules() {
    return ImmutableList.<BlazeModule>builder()
        .addAll(super.getSpawnModules())
        .add(new RemoteModule())
        .build();
  }

  @After
  public void tearDown() throws IOException {
    if (worker != null) {
      worker.stop();
    }
  }

  private void addRemoteModeOptions() throws IOException, InterruptedException {
    if (worker == null) {
      worker = startWorker();
    }

    switch (remoteMode) {
      case REMOTE_EXECUTION:
        addOptions("--remote_executor=grpc://localhost:" + worker.getPort());
        break;
      case REMOTE_CACHE:
        addOptions("--remote_cache=grpc://localhost:" + worker.getPort());
        break;
    }
  }

  private void addOutputModeOptions() {
    switch (outputMode) {
      case DOWNLOAD_TOPLEVEL:
        addOptions("--remote_download_toplevel");
        break;
      case DOWNLOAD_MINIMAL:
        addOptions("--remote_download_minimal");
        break;
    }
  }

  @Test
  public void executeRemotely_unnecessaryOutputsAreNotDownloaded() throws Exception {
    if (!remoteMode.executeRemotely()) {
      return;
    }
    addRemoteModeOptions();
    addOutputModeOptions();
    write(
        "a/BUILD",
        "genrule(",
        "  name = 'foo',",
        "  srcs = [],",
        "  outs = ['foo.txt'],",
        "  cmd = 'echo foo > $@',",
        ")",
        "",
        "genrule(",
        "  name = 'foobar',",
        "  srcs = [':foo'],",
        "  outs = ['foobar.txt'],",
        "  cmd = 'cat $(location :foo) > $@ && echo bar > $@',",
        ")");

    buildTarget("//a:foobar");

    assertOutputsDoNotExist("//a:foo");
    if (outputMode.minimal()) {
      assertOutputsDoNotExist("//a:foobar");
    }
  }

  private void assertOutputsDoNotExist(String target) throws Exception {
    for (Artifact output : getArtifacts(target)) {
      assertThat(output.getPath().exists()).isFalse();
    }
  }

  @Test
  public void executeRemotely_actionFails_outputsAreAvailableLocallyForDebuggingPurpose()
      throws Exception {
    if (!remoteMode.executeRemotely()) {
      return;
    }
    addRemoteModeOptions();
    addOutputModeOptions();
    write(
        "a/BUILD",
        "genrule(",
        "  name = 'fail',",
        "  srcs = [],",
        "  outs = ['fail.txt'],",
        "  cmd = 'echo foo > $@ && exit 1',",
        ")");

    assertThrows(BuildFailedException.class, () -> buildTarget("//a:fail"));

    Artifact output = getOnlyElement(getArtifacts("//a:fail"));
    assertThat(output.getFilename()).isEqualTo("fail.txt");
    assertThat(output.getPath().exists()).isTrue();
    assertThat(readContent(output.getPath(), UTF_8)).isEqualTo("foo\n");
  }

  @Test
  public void
      executeRemotely_intermediateOutputsAreInputForLocalActions_downloadIntermediateOutputs()
          throws Exception {
    // Test that a remotely stored output that's an input to a native action
    // (ctx.actions.expand_template) is staged lazily for action execution.
    if (!remoteMode.executeRemotely()) {
      return;
    }
    addRemoteModeOptions();
    addOutputModeOptions();
    write(
        "a/substitute_username.bzl",
        "def _substitute_username_impl(ctx):",
        "    ctx.actions.expand_template(",
        "        template = ctx.file.template,",
        "        output = ctx.outputs.out,",
        "        substitutions = {",
        "            '{USERNAME}': ctx.attr.username,",
        "        },",
        "    )",
        "",
        "substitute_username = rule(",
        "    implementation = _substitute_username_impl,",
        "    attrs = {",
        "        'username': attr.string(mandatory = True),",
        "        'template': attr.label(",
        "            allow_single_file = True,",
        "            mandatory = True,",
        "        ),",
        "    },",
        "    outputs = {'out': '%{name}.txt'},",
        ")");
    write(
        "a/BUILD",
        "load(':substitute_username.bzl', 'substitute_username')",
        "genrule(",
        "    name = 'generate-template',",
        "    cmd = 'echo -n \"Hello {USERNAME}!\" > $@',",
        "    outs = ['template.txt'],",
        "    srcs = [],",
        ")",
        "",
        "substitute_username(",
        "    name = 'substitute-buchgr',",
        "    username = 'buchgr',",
        "    template = ':generate-template',",
        ")");

    buildTarget("//a:substitute-buchgr");

    // The genrule //a:generate-template should run remotely and //a:substitute-buchgr should be a
    // native action running locally.
    events.assertContainsInfo("3 processes: 2 internal, 1 remote");
    Artifact intermediateOutput = getOnlyElement(getArtifacts("//a:generate-template"));
    assertThat(intermediateOutput.getPath().exists()).isTrue();
    Artifact output = getOnlyElement(getArtifacts("//a:substitute-buchgr"));
    assertThat(output.getFilename()).isEqualTo("substitute-buchgr.txt");
    assertThat(readContent(output.getPath(), UTF_8)).isEqualTo("Hello buchgr!");
  }
}
