// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.sandbox;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.EventBus;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.BlazeExecutor;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.events.PrintingEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestFileOutErr;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import com.google.devtools.common.options.OptionsParser;

import org.junit.Before;

import java.io.IOException;

/**
 * Common parts of all {@link LinuxSandboxedStrategy} tests.
 */
public class LinuxSandboxedStrategyTestCase {
  private Reporter reporter = new Reporter(PrintingEventHandler.ERRORS_AND_WARNINGS_TO_STDERR);
  private Path outputBase;

  protected FileSystem fileSystem;
  protected Path workspaceDir;
  protected Path fakeSandboxDir;

  protected BlazeExecutor executor;
  protected BlazeDirectories blazeDirs;

  protected TestFileOutErr outErr = new TestFileOutErr();

  protected String out() {
    return outErr.outAsLatin1();
  }

  protected String err() {
    return outErr.errAsLatin1();
  }

  @Before
  public final void createDirectoriesAndExecutor() throws Exception  {
    Path testRoot = createTestRoot();

    workspaceDir = testRoot.getRelative("workspace");
    workspaceDir.createDirectory();

    outputBase = testRoot.getRelative("outputBase");
    outputBase.createDirectory();

    fakeSandboxDir = testRoot.getRelative("sandbox");
    fakeSandboxDir.createDirectory();

    blazeDirs = new BlazeDirectories(outputBase, outputBase, workspaceDir,
        TestConstants.PRODUCT_NAME);
    BlazeTestUtils.getIntegrationBinTools(blazeDirs);

    OptionsParser optionsParser =
        OptionsParser.newOptionsParser(ExecutionOptions.class, SandboxOptions.class);

    EventBus bus = new EventBus();

    this.executor =
        new BlazeExecutor(
            blazeDirs.getExecRoot(),
            blazeDirs.getOutputPath(),
            reporter,
            bus,
            BlazeClock.instance(),
            optionsParser,
            /* verboseFailures */ true,
            /* showSubcommands */ false,
            ImmutableList.<ActionContext>of(),
            ImmutableMap.<String, SpawnActionContext>of(
                "",
                new LinuxSandboxedStrategy(
                    optionsParser.getOptions(SandboxOptions.class),
                    ImmutableMap.<String, String>of(),
                    blazeDirs,
                    MoreExecutors.newDirectExecutorService(),
                    true,
                    false,
                    TestConstants.PRODUCT_NAME)),
            ImmutableList.<ActionContextProvider>of());
  }

  protected LinuxSandboxedStrategy getLinuxSandboxedStrategy() {
    SpawnActionContext spawnActionContext = executor.getSpawnActionContext("");
    assertThat(spawnActionContext).isInstanceOf(LinuxSandboxedStrategy.class);
    return (LinuxSandboxedStrategy) spawnActionContext;
  }

  private Path createTestRoot() throws IOException {
    fileSystem = FileSystems.getNativeFileSystem();
    Path testRoot = fileSystem.getPath(TestUtils.tmpDir());
    try {
      FileSystemUtils.deleteTreesBelow(testRoot);
    } catch (IOException e) {
      System.err.println("Failed to remove directory " + testRoot + ": " + e.getMessage());
      throw e;
    }
    return testRoot;
  }
}
