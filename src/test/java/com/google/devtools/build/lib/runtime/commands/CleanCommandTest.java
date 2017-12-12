// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.LockingMode;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlazeServerStartupOptions;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.common.options.OptionsParser;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link CleanCommand}.
 */
@RunWith(JUnit4.class)
public class CleanCommandTest {

  private final RecordingOutErr outErr = new RecordingOutErr();
  private Scratch scratch = new Scratch();
  private BlazeRuntime runtime;
  private BlazeCommand command;
  private BlazeCommandDispatcher dispatcher;

  @Before
  public final void initializeRuntime() throws Exception {
    String productName = TestConstants.PRODUCT_NAME;
    ServerDirectories serverDirectories =
        new ServerDirectories(scratch.dir("install"), scratch.dir("output"));
    this.runtime =
        new BlazeRuntime.Builder()
            .setFileSystem(scratch.getFileSystem())
            .setProductName(productName)
            .setServerDirectories(serverDirectories)
            .setStartupOptionsProvider(
                OptionsParser.newOptionsParser(BlazeServerStartupOptions.class))
            .addBlazeModule(
                new BlazeModule() {
                  @Override
                  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
                    // We must add these options so that the defaults package can be created.
                    builder.addConfigurationOptions(BuildConfiguration.Options.class);
                    // The tools repository is needed for createGlobals
                    builder.setToolsRepository(TestConstants.TOOLS_REPOSITORY);
                  }
                })
            .build();
    BlazeDirectories directories =
        new BlazeDirectories(serverDirectories, scratch.dir("workspace"), productName);
    this.runtime.initWorkspace(directories, /* binTools= */ null);
    this.command = new CleanCommand();
    this.dispatcher = new BlazeCommandDispatcher(this.runtime, this.command);
  }

  @Test
  public void testCleanWithAsyncDoesNotSuggestAsync() throws Exception {
    List<String> commandLine = Lists.newArrayList("clean", "--async");
    this.dispatcher.exec(commandLine, LockingMode.ERROR_OUT, "test", outErr);
    String output = outErr.toString();
    String suggestion = "Consider using --async";
    assertWithMessage("clean --async command shouldn't suggest using --async")
        .that(output).doesNotContain(suggestion);
  }

  @Test
  public void testCleanSuggestsAsyncOnLinuxPlatformsOnly() throws Exception {
    List<String> commandLine = Lists.newArrayList("clean");
    this.dispatcher.exec(commandLine, LockingMode.ERROR_OUT, "test", outErr);
    String output = outErr.toString();
    String suggestion = "Consider using --async";
    if (OS.getCurrent() == OS.LINUX) {
      assertWithMessage("clean command should suggest using --async on Linux platforms")
          .that(output).contains(suggestion);
    } else {
      assertWithMessage("clean command shouldn't suggest using --async on non-Linux platforms")
          .that(output).doesNotContain(suggestion);
    }
  }
}
