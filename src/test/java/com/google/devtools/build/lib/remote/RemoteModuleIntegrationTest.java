// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.remote.options.RemoteStartupOptions;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils.WorkerInstance;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.common.options.OptionsBase;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.After;
import org.junit.ClassRule;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteModule} across commands in the same server. */
@RunWith(JUnit4.class)
public final class RemoteModuleIntegrationTest extends BuildIntegrationTestCase {
  @ClassRule @Rule public static final WorkerInstance worker = IntegrationTestUtils.createWorker();

  private final RemoteModule remoteModule = new RemoteModule();

  @Override
  protected ImmutableList<Class<? extends OptionsBase>> getStartupOptionClasses() {
    return ImmutableList.<Class<? extends OptionsBase>>builder()
        .addAll(super.getStartupOptionClasses())
        .add(RemoteStartupOptions.class)
        .build();
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(remoteModule)
        .addBlazeModule(new BlockWaitingModule());
  }

  @After
  public void shutdownExecutor() {
    remoteModule.getExecutorService().shutdownNow();
  }

  @Test
  public void executorThreads_doNotAccumulateAcrossCommands() throws Exception {
    addOptions("--remote_cache=grpc://localhost:" + worker.getPort(), "--jobs=1");
    write("BUILD", "genrule(name = 'copy', srcs = ['input'], outs = ['output'], cmd = 'cp $< $@')");
    var threadsBefore = remoteExecutorThreads();

    for (int i = 0; i < 3; i++) {
      // Force a new action and cache traffic on each command.
      write("input", "build " + i);
      buildTarget("//:copy");

      var threads = remoteExecutorThreads();
      threads.removeAll(threadsBefore);
      // Count live threads across all pools, including pools abandoned by previous commands.
      // Requiring one thread also verifies that the build actually exercised the executor.
      assertThat(threads).hasSize(1);
    }
  }

  private static Set<Thread> remoteExecutorThreads() {
    return Thread.getAllStackTraces().keySet().stream()
        .filter(thread -> thread.getName().startsWith("remote-executor-"))
        .collect(Collectors.toSet());
  }
}
