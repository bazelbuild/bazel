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
package com.google.devtools.build.lib.standalone;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.BlazeExecutor;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.events.PrintingEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SingleBuildFileCache;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.testutil.TestFileOutErr;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import com.google.devtools.common.options.OptionsParser;
import java.io.IOException;
import java.util.Arrays;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test StandaloneSpawnStrategy.
 */
@RunWith(JUnit4.class)
public class StandaloneSpawnStrategyTest {

  private Reporter reporter = new Reporter(PrintingEventHandler.ERRORS_AND_WARNINGS_TO_STDERR);
  private BlazeExecutor executor;
  private FileSystem fileSystem;

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

  @Before
  public final void setUp() throws Exception {
    Path testRoot = createTestRoot();
    Path workspaceDir = testRoot.getRelative("workspace-name");
    workspaceDir.createDirectory();

    // setup output base & directories
    Path outputBase = testRoot.getRelative("outputBase");
    outputBase.createDirectory();

    BlazeDirectories directories =
        new BlazeDirectories(outputBase, outputBase, workspaceDir, "mock-product-name");
    BlazeTestUtils.getIntegrationBinTools(directories);
    OptionsParser optionsParser = OptionsParser.newOptionsParser(ExecutionOptions.class);
    optionsParser.parse("--verbose_failures");

    EventBus bus = new EventBus();

    this.executor =
        new BlazeExecutor(
            directories.getExecRoot(),
            reporter,
            bus,
            BlazeClock.instance(),
            optionsParser,
            /* verboseFailures */ false,
            /* showSubcommands */ false,
            ImmutableList.<ActionContext>of(),
            ImmutableMap.<String, SpawnActionContext>of(
                "",
                new StandaloneSpawnStrategy(directories.getExecRoot(), false, "mock-product-name")),
            ImmutableList.<ActionContextProvider>of());

    executor.getExecRoot().createDirectory();
  }

  private Spawn createSpawn(String... arguments) {
    return new BaseSpawn.Local(Arrays.asList(arguments), ImmutableMap.<String, String>of(),
        new ActionsTestUtil.NullAction());
  }

  private TestFileOutErr outErr = new TestFileOutErr();

  private String out() {
    return outErr.outAsLatin1();
  }
  private String err() {
    return outErr.errAsLatin1();
  }

  @Test
  public void testBinTrueExecutesFine() throws Exception {
    Spawn spawn = createSpawn(getTrueCommand());
    executor.getSpawnActionContext(spawn.getMnemonic()).exec(spawn, createContext());

    assertThat(out()).isEmpty();
    assertThat(err()).isEmpty();
  }

  private void run(Spawn spawn) throws Exception {
    executor.getSpawnActionContext(spawn.getMnemonic()).exec(spawn, createContext());
  }
  private ActionExecutionContext createContext() {
    Path execRoot = executor.getExecRoot();
    return new ActionExecutionContext(
        executor,
        new SingleBuildFileCache(execRoot.getPathString(), execRoot.getFileSystem()),
        null,
        outErr,
        ImmutableMap.<String, String>of(),
        null);
  }

  @Test
  public void testBinFalseYieldsException() throws Exception {
    try {
      run(createSpawn(getFalseCommand()));
      fail();
    } catch (ExecException e) {
      assertTrue("got: " + e.getMessage(), e
          .getMessage().startsWith("false failed: error executing command"));
    }
  }

  private static String getFalseCommand() {
    return OS.getCurrent() == OS.DARWIN ? "/usr/bin/false" : "/bin/false";
  }

  private static String getTrueCommand() {
    return OS.getCurrent() == OS.DARWIN ? "/usr/bin/true" : "/bin/true";
  }

  @Test
  public void testBinEchoPrintsArguments() throws Exception {
    Spawn spawn = createSpawn("/bin/echo", "Hello,", "world.");
    run(spawn);
    assertEquals("Hello, world.\n", out());
    assertThat(err()).isEmpty();
  }

  @Test
  public void testCommandRunsInWorkingDir() throws Exception {
    Spawn spawn = createSpawn("/bin/pwd");
    run(spawn);
    assertEquals(executor.getExecRoot() + "\n", out());
  }

  @Test
  public void testCommandHonorsEnvironment() throws Exception {
    Spawn spawn = new BaseSpawn.Local(Arrays.asList("/usr/bin/env"),
        ImmutableMap.of("foo", "bar", "baz", "boo"),
        new ActionsTestUtil.NullAction());
    run(spawn);
    assertEquals(Sets.newHashSet("foo=bar", "baz=boo"), Sets.newHashSet(out().split("\n")));
  }

  @Test
  public void testStandardError() throws Exception {
    Spawn spawn = createSpawn("/bin/sh", "-c", "echo Oops! >&2");
    run(spawn);
    assertEquals("Oops!\n", err());
    assertThat(out()).isEmpty();
  }

  // Test an action with environment variables set indicating an action running on a darwin host
  // system. Such actions should fail given the fact that these tests run on a non darwin
  // architecture.
  @Test
  public void testIOSEnvironmentOnNonDarwin() throws Exception {
    if (OS.getCurrent() == OS.DARWIN) {
      return;
    }
    Spawn spawn = new BaseSpawn.Local(Arrays.asList("/bin/sh", "-c", "echo $SDKROOT"),
        ImmutableMap.<String, String>of(AppleConfiguration.APPLE_SDK_VERSION_ENV_NAME, "8.4",
            AppleConfiguration.APPLE_SDK_PLATFORM_ENV_NAME, "iPhoneSimulator"),
        new ActionsTestUtil.NullAction());

    try {
      run(spawn);
      fail("action should fail due to being unable to resolve SDKROOT");
    } catch (ExecException e) {
      assertThat(e.getMessage()).contains("Cannot locate iOS SDK on non-darwin operating system");
    }
  }
}
