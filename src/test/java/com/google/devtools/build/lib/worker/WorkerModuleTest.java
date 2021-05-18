// Copyright 2021 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.worker;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.ExecutionRequirements.WorkerProtocolFormat.JSON;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Tests for WorkerModule. I bet you didn't see that coming, eh? */
@RunWith(JUnit4.class)
public class WorkerModuleTest {
  @Rule public final MockitoRule mockito = MockitoJUnit.rule();
  @Mock CommandEnvironment env;
  @Mock BuildRequest request;
  @Mock OptionsParsingResult startupOptionsProvider;

  private FileSystem fs;
  private StoredEventHandler storedEventHandler;

  @Before
  public void setUp() {
    fs = new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
  }

  @Test
  public void buildStarting_createsPools()
      throws AbruptExitException, IOException, InterruptedException {
    WorkerModule module = new WorkerModule();
    WorkerOptions options = WorkerOptions.DEFAULTS;
    when(request.getOptions(WorkerOptions.class)).thenReturn(options);
    setupEnvironment("/outputRoot");

    module.beforeCommand(env);
    module.buildStarting(new BuildStartingEvent(env, request));
    assertThat(storedEventHandler.getEvents()).isEmpty();
    assertThat(fs.getPath("/outputRoot/outputBase/bazel-workers").exists()).isTrue();
    assertThat(module.workerPool).isNotNull();
    WorkerKey workerKey = TestUtils.createWorkerKey(JSON, fs);
    Worker worker = module.workerPool.borrowObject(workerKey);
    assertThat(worker.workerKey).isEqualTo(workerKey);
  }

  @Test
  public void buildStarting_restartsOnSandboxChanges() throws IOException, AbruptExitException {
    WorkerModule module = new WorkerModule();
    WorkerOptions options = WorkerOptions.DEFAULTS;
    when(request.getOptions(WorkerOptions.class)).thenReturn(options);
    setupEnvironment("/outputRoot");

    module.beforeCommand(env);
    module.buildStarting(new BuildStartingEvent(env, request));
    assertThat(storedEventHandler.getEvents()).isEmpty();

    Path workerDir = fs.getPath("/outputRoot/outputBase/bazel-workers");
    Path aLog = workerDir.getRelative("f.log");
    aLog.createSymbolicLink(PathFragment.EMPTY_FRAGMENT);
    WorkerPool oldPool = module.workerPool;
    options.workerSandboxing = !options.workerSandboxing;
    module.beforeCommand(env);
    module.buildStarting(new BuildStartingEvent(env, request));
    assertThat(storedEventHandler.getEvents()).hasSize(1);
    assertThat(storedEventHandler.getEvents().get(0).getMessage())
        .contains("Worker factory configuration has changed");
    assertThat(module.workerPool).isNotSameInstanceAs(oldPool);
    assertThat(aLog.exists()).isFalse();
  }

  @Test
  public void buildStarting_workersDestroyedOnRestart()
      throws IOException, AbruptExitException, InterruptedException {
    WorkerModule module = new WorkerModule();
    WorkerOptions options = WorkerOptions.DEFAULTS;
    options.workerVerbose = true;
    when(request.getOptions(WorkerOptions.class)).thenReturn(options);
    setupEnvironment("/outputRoot");

    module.beforeCommand(env);
    module.buildStarting(new BuildStartingEvent(env, request));
    WorkerKey workerKey = TestUtils.createWorkerKey(JSON, fs);
    Worker worker = module.workerPool.borrowObject(workerKey);
    assertThat(worker.workerKey).isEqualTo(workerKey);
    assertThat(storedEventHandler.getEvents()).hasSize(1);
    assertThat(storedEventHandler.getEvents().get(0).getMessage())
        .contains("Created new sandboxed dummy worker");
    storedEventHandler.clear();

    Path workerDir = fs.getPath("/outputRoot/outputBase/bazel-workers");
    Path aLog = workerDir.getRelative("f.log");
    aLog.createSymbolicLink(PathFragment.EMPTY_FRAGMENT);
    WorkerPool oldPool = module.workerPool;
    options.workerSandboxing = !options.workerSandboxing;
    module.beforeCommand(env);
    module.buildStarting(new BuildStartingEvent(env, request));
    assertThat(storedEventHandler.getEvents()).hasSize(1);
    assertThat(storedEventHandler.getEvents().get(0).getMessage())
        .contains("Worker factory configuration has changed");
    assertThat(module.workerPool).isNotSameInstanceAs(oldPool);
    assertThat(aLog.exists()).isFalse();
  }

  @Test
  public void buildStarting_restartsOnOutputbaseChanges() throws IOException, AbruptExitException {
    WorkerModule module = new WorkerModule();
    WorkerOptions options = WorkerOptions.DEFAULTS;
    when(request.getOptions(WorkerOptions.class)).thenReturn(options);
    setupEnvironment("/outputRoot");

    module.beforeCommand(env);
    module.buildStarting(new BuildStartingEvent(env, request));
    assertThat(storedEventHandler.getEvents()).isEmpty();

    // Log file from old root, doesn't get cleaned
    Path workerDir = fs.getPath("/outputRoot/outputBase/bazel-workers");
    Path oldLog = workerDir.getRelative("f.log");
    oldLog.createSymbolicLink(PathFragment.EMPTY_FRAGMENT);

    WorkerPool oldPool = module.workerPool;
    setupEnvironment("/otherRootDir");
    module.beforeCommand(env);
    module.buildStarting(new BuildStartingEvent(env, request));
    assertThat(storedEventHandler.getEvents()).hasSize(1);
    assertThat(storedEventHandler.getEvents().get(0).getMessage())
        .contains("Worker factory configuration has changed");
    assertThat(module.workerPool).isNotSameInstanceAs(oldPool);
    assertThat(fs.getPath("/otherRootDir/outputBase/bazel-workers").exists()).isTrue();
    assertThat(oldLog.exists()).isTrue();
  }

  @Test
  public void buildStarting_restartsOnHiPrioChanges() throws IOException, AbruptExitException {
    WorkerModule module = new WorkerModule();
    WorkerOptions options = WorkerOptions.DEFAULTS;
    when(request.getOptions(WorkerOptions.class)).thenReturn(options);
    setupEnvironment("/outputRoot");

    module.beforeCommand(env);
    // Check that new pools/factories are made with default options
    module.buildStarting(new BuildStartingEvent(env, request));
    assertThat(storedEventHandler.getEvents()).isEmpty();

    // Logs are only cleared on factory reset, not on pool reset, so this file should survive
    Path workerDir = fs.getPath("/outputRoot/outputBase/bazel-workers");
    Path oldLog = workerDir.getRelative("f.log");
    oldLog.createSymbolicLink(PathFragment.EMPTY_FRAGMENT);

    WorkerPool oldPool = module.workerPool;
    options.highPriorityWorkers = ImmutableList.of("foo");
    module.beforeCommand(env);
    module.buildStarting(new BuildStartingEvent(env, request));
    assertThat(storedEventHandler.getEvents()).hasSize(1);
    assertThat(storedEventHandler.getEvents().get(0).getMessage())
        .contains("Worker pool configuration has changed");
    assertThat(module.workerPool).isNotSameInstanceAs(oldPool);
    assertThat(oldLog.exists()).isTrue();
  }

  @Test
  public void buildStarting_restartsOnNumMultiplexWorkersChanges()
      throws IOException, AbruptExitException {
    WorkerModule module = new WorkerModule();
    WorkerOptions options = WorkerOptions.DEFAULTS;
    when(request.getOptions(WorkerOptions.class)).thenReturn(options);
    setupEnvironment("/outputRoot");

    module.beforeCommand(env);
    // Check that new pools/factories are made with default options
    module.buildStarting(new BuildStartingEvent(env, request));
    assertThat(storedEventHandler.getEvents()).isEmpty();

    WorkerPool oldPool = module.workerPool;
    options.workerMaxMultiplexInstances = Lists.newArrayList(Maps.immutableEntry("foo", 42));
    module.beforeCommand(env);
    module.buildStarting(new BuildStartingEvent(env, request));
    assertThat(storedEventHandler.getEvents()).hasSize(1);
    assertThat(storedEventHandler.getEvents().get(0).getMessage())
        .contains("Worker pool configuration has changed");
    assertThat(module.workerPool).isNotSameInstanceAs(oldPool);
  }

  @Test
  public void buildStarting_restartsOnNumWorkersChanges() throws IOException, AbruptExitException {
    WorkerModule module = new WorkerModule();
    WorkerOptions options = WorkerOptions.DEFAULTS;

    when(request.getOptions(WorkerOptions.class)).thenReturn(options);
    setupEnvironment("/outputRoot");

    module.beforeCommand(env);
    // Check that new pools/factories are made with default options
    module.buildStarting(new BuildStartingEvent(env, request));
    assertThat(storedEventHandler.getEvents()).isEmpty();

    WorkerPool oldPool = module.workerPool;
    options.workerMaxInstances = Lists.newArrayList(Maps.immutableEntry("bar", 3));
    module.beforeCommand(env);
    module.buildStarting(new BuildStartingEvent(env, request));
    assertThat(storedEventHandler.getEvents()).hasSize(1);
    assertThat(storedEventHandler.getEvents().get(0).getMessage())
        .contains("Worker pool configuration has changed");
    assertThat(module.workerPool).isNotSameInstanceAs(oldPool);
  }

  private void setupEnvironment(String rootDir) throws IOException, AbruptExitException {
    storedEventHandler = new StoredEventHandler();
    Path root = fs.getPath(rootDir);
    Path outputBase = root.getRelative("outputBase");
    outputBase.createDirectoryAndParents();
    when(env.getOutputBase()).thenReturn(outputBase);
    Path workspace = fs.getPath("/workspace");
    when(env.getWorkingDirectory()).thenReturn(workspace);
    ServerDirectories serverDirectories =
        new ServerDirectories(
            root.getRelative("userroot/install"), outputBase, root.getRelative("userroot"));
    BlazeRuntime blazeRuntime =
        new BlazeRuntime.Builder()
            .setProductName("bazel")
            .setServerDirectories(serverDirectories)
            .setStartupOptionsProvider(startupOptionsProvider)
            .build();
    when(env.getRuntime()).thenReturn(blazeRuntime);
    when(env.getDirectories())
        .thenReturn(new BlazeDirectories(serverDirectories, null, null, "blaze"));
    EventBus eventBus = new EventBus();
    when(env.getEventBus()).thenReturn(eventBus);
    when(env.getReporter()).thenReturn(new Reporter(eventBus, storedEventHandler));
  }
}
