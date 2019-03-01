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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.bazel.rules.DefaultBuildOptionsForDiffing;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlazeServerStartupOptions;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.common.options.OptionsParser;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/** Tests {@link CleanCommand}'s recommendation of the --async flag. */
@RunWith(Parameterized.class)
public class CleanCommandRecommendsAsyncTest {

  private final RecordingOutErr outErr = new RecordingOutErr();
  private final List<String> commandLine;
  private final OS os;
  private final boolean expectSuggestion;
  private static final String EXPECTED_SUGGESTION = "Consider using --async";

  public CleanCommandRecommendsAsyncTest(List<String> commandLine, OS os, boolean expectSuggestion)
      throws Exception {
    this.commandLine = commandLine;
    this.os = os;
    this.expectSuggestion = expectSuggestion;
  }

  @Parameters(name = "Command line {0} on OS {1}")
  public static Iterable<Object[]> data() {
    return Arrays.asList(
        new Object[][] {
          // When --async is provided, don't expect --async to be suggested.
          {ImmutableList.of("clean", "--async"), OS.LINUX, false},
          {ImmutableList.of("clean", "--async"), OS.WINDOWS, false},
          {ImmutableList.of("clean", "--async"), OS.DARWIN, false},

          // When --async is not provided, expect the suggestion on platforms that support it.
          {ImmutableList.of("clean"), OS.LINUX, true},
          {ImmutableList.of("clean"), OS.WINDOWS, false},
          {ImmutableList.of("clean"), OS.DARWIN, false},

          // When --noasync is explicitly provided, unfortunately we still expect the suggestion,
          // since there's no way to tell the difference between false-by-default and explicit
          // false.
          {ImmutableList.of("clean", "--noasync"), OS.LINUX, true},
        });
  }

  @Test
  public void testCleanProvidesExpectedSuggestion() throws Exception {
    String productName = TestConstants.PRODUCT_NAME;
    Scratch scratch = new Scratch();
    ServerDirectories serverDirectories =
        new ServerDirectories(
            scratch.dir("install"), scratch.dir("output"), scratch.dir("user_root"));
    BlazeRuntime runtime =
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
                    builder.addConfigurationOptions(TestConfiguration.TestOptions.class);
                    // The tools repository is needed for createGlobals
                    builder.setToolsRepository(TestConstants.TOOLS_REPOSITORY);
                  }
                })
            .addBlazeModule(
                new BlazeModule() {
                  @Override
                  public BuildOptions getDefaultBuildOptions(BlazeRuntime runtime) {
                    return DefaultBuildOptionsForDiffing.getDefaultBuildOptionsForFragments(
                        runtime.getRuleClassProvider().getConfigurationOptions());
                  }
                })
            .build();
    BlazeDirectories directories =
        new BlazeDirectories(
            serverDirectories,
            scratch.dir("workspace"),
            /* defaultSystemJavabase= */ null,
            productName);
    runtime.initWorkspace(directories, /* binTools= */ null);

    BlazeCommandDispatcher dispatcher = new BlazeCommandDispatcher(runtime, new CleanCommand(os));
    dispatcher.exec(commandLine, "test", outErr);
    String output = outErr.toString();

    if (expectSuggestion) {
      assertThat(output).contains(EXPECTED_SUGGESTION);
    } else {
      assertThat(output).doesNotContain(EXPECTED_SUGGESTION);
    }
  }
}
