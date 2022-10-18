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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.remote.util.IntegrationTestUtils.startWorker;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.readContent;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionExecutedEvent;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.CachedActionEvent;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils.WorkerInstance;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.runtime.BuildSummaryStatsModule;
import com.google.devtools.build.lib.sandbox.SandboxModule;
import com.google.devtools.build.lib.util.OS;
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
    return super.getRuntimeBuilder()
        .addBlazeModule(new BuildSummaryStatsModule())
        .addBlazeModule(new BlockWaitingModule());
  }

  @Override
  protected ImmutableList<BlazeModule> getSpawnModules() {
    return ImmutableList.<BlazeModule>builder()
        .addAll(super.getSpawnModules())
        .add(new SandboxModule())
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
    addRemoteModeOptions(remoteMode);
  }

  private void addRemoteModeOptions(RemoteMode remoteMode)
      throws IOException, InterruptedException {
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

  private String getSpawnStrategy() {
    if (remoteMode.executeRemotely()) {
      return "remote";
    } else {
      OS currentOS = OS.getCurrent();
      switch (currentOS) {
        case LINUX:
          return "linux-sandbox";
        case DARWIN:
          return "processwrapper-sandbox";
        default:
          return "local";
      }
    }
  }

  @Test
  public void unnecessaryOutputsAreNotDownloaded() throws Exception {
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
    if (!remoteMode.executeRemotely()) {
      warmUpRemoteCache("//a:foobar");
    }
    addRemoteModeOptions();
    addOutputModeOptions();

    buildTarget("//a:foobar");

    assertOutputsDoNotExist("//a:foo");
    if (outputMode.minimal()) {
      assertOutputsDoNotExist("//a:foobar");
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

    assertOnlyOutputContent("//a:fail", "fail.txt", "foo\n");
  }

  @Test
  public void intermediateOutputsAreInputForLocalActions_prefetchIntermediateOutputs()
      throws Exception {
    // Test that a remote-only output that's an input to a local action is downloaded lazily before
    // executing the local action.
    write(
        "a/BUILD",
        "genrule(",
        "  name = 'remote',",
        "  srcs = [],",
        "  outs = ['remote.txt'],",
        "  cmd = 'echo -n remote > $@',",
        ")",
        "",
        "genrule(",
        "  name = 'local',",
        "  srcs = [':remote'],",
        "  outs = ['local.txt'],",
        "  cmd = 'cat $(location :remote) > $@ && echo -n local >> $@',",
        "  tags = ['no-remote'],",
        ")");
    if (!remoteMode.executeRemotely()) {
      warmUpRemoteCache("//a:local");
    }
    addRemoteModeOptions();
    addOutputModeOptions();

    if (outputMode.minimal()) {
      buildTarget("//a:remote");
      assertOutputsDoNotExist("//a:remote");
    }
    buildTarget("//a:local");

    assertOnlyOutputContent("//a:remote", "remote.txt", "remote");
    assertOnlyOutputContent("//a:local", "local.txt", "remotelocal");
  }

  @Test
  public void intermediateOutputsAreInputForInternalActions_prefetchIntermediateOutputs()
      throws Exception {
    // Disable on Windows since it seems that template is not supported there.
    if (OS.getCurrent() == OS.WINDOWS) {
      return;
    }
    // Test that a remotely stored output that's an input to a internal action
    // (ctx.actions.expand_template) is staged lazily for action execution.
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
    if (!remoteMode.executeRemotely()) {
      warmUpRemoteCache("//a:substitute-buchgr");
    }
    addRemoteModeOptions();
    addOutputModeOptions();

    buildTarget("//a:substitute-buchgr");

    // The genrule //a:generate-template should run remotely and //a:substitute-buchgr should be a
    // internal action running locally.
    if (remoteMode.executeRemotely()) {
      events.assertContainsInfo("3 processes: 2 internal, 1 remote");
    } else {
      events.assertContainsInfo("3 processes: 1 remote cache hit, 2 internal");
    }
    Artifact intermediateOutput = getOnlyElement(getArtifacts("//a:generate-template"));
    assertThat(intermediateOutput.getPath().exists()).isTrue();
    assertOnlyOutputContent("//a:substitute-buchgr", "substitute-buchgr.txt", "Hello buchgr!");
  }

  @Test
  public void changeOutputMode_invalidateActions() throws Exception {
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
    ActionEventCollector actionEventCollector = new ActionEventCollector();
    runtimeWrapper.registerSubscriber(actionEventCollector);
    buildTarget("//a:foobar");
    // 3 = workspace status action + //:foo + //:foobar
    assertThat(actionEventCollector.getNumActionNodesEvaluated()).isEqualTo(3);
    actionEventCollector.clear();
    events.assertContainsInfo("3 processes: 1 internal, 2 " + getSpawnStrategy());
    events.clear();

    addOptions("--remote_download_outputs=all");
    buildTarget("//a:foobar");

    // Changing output mode should invalidate SkyFrame's in-memory caching and make it re-evaluate
    // the action nodes.
    assertThat(actionEventCollector.getNumActionNodesEvaluated()).isEqualTo(3);
    if (remoteMode.executeRemotely()) {
      // The dummy workspace status action does not re-executed, so wouldn't be displayed here
      if (outputMode.minimal()) {
        events.assertContainsInfo("2 processes: 2 remote cache hit");
      } else {
        // output of toplevel target are on the local disk, so the action can hit the action cache,
        // hence not shown here
        events.assertContainsInfo("1 process: 1 remote cache hit");
      }
    } else {
      // all actions can hit action cache since all outputs are on the local disk due to they were
      // executed locally.
      events.assertContainsInfo("0 processes");
    }
  }

  private static class ActionEventCollector {
    private final List<ActionExecutedEvent> actionExecutedEvents = new ArrayList<>();
    private final List<CachedActionEvent> cachedActionEvents = new ArrayList<>();

    @Subscribe
    public void onActionExecuted(ActionExecutedEvent event) {
      actionExecutedEvents.add(event);
    }

    @Subscribe
    public void onCachedAction(CachedActionEvent event) {
      cachedActionEvents.add(event);
    }

    public int getNumActionNodesEvaluated() {
      return getActionExecutedEvents().size() + getCachedActionEvents().size();
    }

    public void clear() {
      this.actionExecutedEvents.clear();
      this.cachedActionEvents.clear();
    }

    public List<ActionExecutedEvent> getActionExecutedEvents() {
      return actionExecutedEvents;
    }

    public List<CachedActionEvent> getCachedActionEvents() {
      return cachedActionEvents;
    }
  }

  private void assertOutputsDoNotExist(String target) throws Exception {
    for (Artifact output : getArtifacts(target)) {
      assertWithMessage(
              "output %s for target %s should not exist", output.getExecPathString(), target)
          .that(output.getPath().exists())
          .isFalse();
    }
  }

  private void assertOnlyOutputContent(String target, String filename, String content)
      throws Exception {
    Artifact output = getOnlyElement(getArtifacts(target));
    assertThat(output.getFilename()).isEqualTo(filename);
    assertThat(output.getPath().exists()).isTrue();
    assertThat(readContent(output.getPath(), UTF_8)).isEqualTo(content);
  }

  private void warmUpRemoteCache(String target) throws Exception {
    checkState(worker == null, "Call 'warmUpRemoteCache' before 'addRemoteModeOptions'");
    addRemoteModeOptions(RemoteMode.REMOTE_EXECUTION);
    ImmutableList<String> originalOptions = getRuntimeWrapper().getOptions();

    buildTarget(target);

    getDirectories().getOutputPath(getWorkspace().getBaseName()).deleteTreesBelow();
    resetOptions();
    addOptions(originalOptions);
  }
}
