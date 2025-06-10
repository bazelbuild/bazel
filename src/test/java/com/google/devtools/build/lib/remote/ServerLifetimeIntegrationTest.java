// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionExecutedEvent;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CachedActionEvent;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.dynamic.DynamicExecutionModule;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils.WorkerInstance;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.runtime.BuildSummaryStatsModule;
import com.google.devtools.build.lib.standalone.StandaloneModule;
import java.util.ArrayList;
import java.util.List;
import org.junit.ClassRule;
import org.junit.Rule;
import org.junit.Test;

/** Integration tests for --experimental_remote_cache_ttl=server. */
public class ServerLifetimeIntegrationTest extends BuildIntegrationTestCase {
  @ClassRule @Rule public static final WorkerInstance worker = IntegrationTestUtils.createWorker();

  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();

    addOptions(
        "--remote_executor=grpc://localhost:" + worker.getPort(),
        "--remote_download_minimal",
        "--dynamic_local_strategy=standalone",
        "--dynamic_remote_strategy=remote");
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(new RemoteModule())
        .addBlazeModule(new BuildSummaryStatsModule())
        .addBlazeModule(new BlockWaitingModule());
  }

  @Override
  protected ImmutableList<BlazeModule> getSpawnModules() {
    return ImmutableList.<BlazeModule>builder()
        .addAll(super.getSpawnModules())
        .add(new StandaloneModule())
        .add(new CredentialModule())
        .add(new DynamicExecutionModule())
        .build();
  }

  protected void assertOutputsDoNotExist(String target) throws Exception {
    for (Artifact output : getArtifacts(target)) {
      assertWithMessage(
              "output %s for target %s should not exist", output.getExecPathString(), target)
          .that(output.getPath().exists())
          .isFalse();
    }
  }

  protected static class ActionEventCollector {
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

  protected void restartServer() throws Exception {
    // Simulates a server restart
    createRuntimeWrapper();
  }

  @Test
  public void incrementalBuild_restartServer_missActionCache() throws Exception {
    // Prepare workspace
    addOptions("--experimental_remote_cache_ttl=server");
    write(
        "BUILD",
        "genrule(",
        "  name = 'foo',",
        "  srcs = [],",
        "  outs = ['out/foo.txt'],",
        "  cmd = 'echo foo > $@',",
        ")",
        "genrule(",
        "  name = 'foobar',",
        "  srcs = [':foo'],",
        "  outs = ['out/foobar.txt'],",
        "  cmd = 'cat $(location :foo) > $@ && echo bar >> $@',",
        ")");
    ActionEventCollector actionEventCollector = new ActionEventCollector();
    getRuntimeWrapper().registerSubscriber(actionEventCollector);

    // Clean build
    buildTarget("//:foobar");

    // all action should be executed
    assertThat(actionEventCollector.getActionExecutedEvents()).hasSize(3);
    // no outputs are staged
    assertOutputsDoNotExist("//:foobar");

    restartServer();
    actionEventCollector = new ActionEventCollector();
    getRuntimeWrapper().registerSubscriber(actionEventCollector);

    // Incremental build
    buildTarget("//:foobar");

    // We expect to see actions here due to expiring the server-lifetime entries
    // in the action cache.
    assertThat(actionEventCollector.getActionExecutedEvents()).hasSize(2);
    // no outputs are staged
    assertOutputsDoNotExist("//:foobar");
  }
}
