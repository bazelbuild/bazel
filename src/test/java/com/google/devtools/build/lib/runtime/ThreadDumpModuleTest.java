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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.BazelServices.BAZEL_SERVICES;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.profiler.MemoryProfiler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link ThreadDumpModule}. */
@RunWith(TestParameterInjector.class)
public final class ThreadDumpModuleTest {
  private final Scratch scratch = new Scratch();
  private final RecordingOutErr outErr = new RecordingOutErr();
  private BlazeCommandDispatcher dispatcher;

  @Command(name = "noop", shortDescription = "", help = "")
  private static final class NoopCommand implements BlazeCommand {
    @Override
    public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
      return BlazeCommandResult.success();
    }
  }

  @Before
  public void setUp() throws Exception {
    String productName = TestConstants.PRODUCT_NAME;
    ServerDirectories serverDirectories =
        new ServerDirectories(
            scratch.dir("install_base"), scratch.dir("output_base"), scratch.dir("user_root"));
    OptionsParsingResult startupOptionsProvider =
        OptionsParser.builder().optionsClasses(BlazeServerStartupOptions.class).build();
    for (BlazeService service : BAZEL_SERVICES) {
      service.globalInit(startupOptionsProvider, BAZEL_SERVICES);
    }
    BlazeRuntime.Builder builder =
        new BlazeRuntime.Builder()
            .setFileSystem(scratch.getFileSystem())
            .setServerDirectories(serverDirectories)
            .setProductName(productName)
            .setStartupOptionsProvider(startupOptionsProvider)
            .addBlazeModule(new ThreadDumpModule());
    for (BlazeService service : BAZEL_SERVICES) {
      builder.addBlazeService(service);
    }
    BlazeRuntime runtime = builder.build();
    runtime.initWorkspace(
        new BlazeDirectories(serverDirectories, scratch.dir("workspace"), productName),
        /* binTools= */ null);
    runtime.overrideCommands(ImmutableList.of(new NoopCommand()));
    dispatcher = new BlazeCommandDispatcher(runtime);
  }

  @After
  public void tearDown() throws Exception {
    try {
      Profiler.instance().stop();
      MemoryProfiler.instance().stop();
    } finally {
      Profiler.setTraceProfilerService(null);
      BugReport.maybePropagateLastCrashIfInTest();
    }
  }

  @Test
  public void exec_subSecondInterval_returnsCommandLineError(
      @TestParameter({"1ns", "1ms", "999ms"}) String interval) throws Exception {
    BlazeCommandResult result =
        dispatcher.exec(
            ImmutableList.of(
                "noop",
                "--experimental_enable_thread_dump",
                "--experimental_thread_dump_interval=" + interval),
            "test",
            outErr);

    assertThat(result.getExitCode()).isEqualTo(ExitCode.COMMAND_LINE_ERROR);
    assertThat(result.getDetailedExitCode().getFailureDetail().getCommand().getCode())
        .isEqualTo(FailureDetails.Command.Code.OPTIONS_PARSE_FAILURE);
    assertThat(outErr.errAsLatin1())
        .contains(
            "ERROR: While parsing option --experimental_thread_dump_interval="
                + interval
                + ": Duration '"
                + interval
                + "' must be zero or at least 1s.");
  }

  @Test
  public void exec_subSecondIntervalWithUnrecognizedFlag_doesNotCrash() throws Exception {
    BlazeCommandResult result =
        dispatcher.exec(
            ImmutableList.of(
                "noop",
                "--experimental_enable_thread_dump",
                "--experimental_thread_dump_interval=1ns",
                "--bogus_invalid_flag"),
            "test",
            outErr);

    assertThat(result.getExitCode()).isEqualTo(ExitCode.COMMAND_LINE_ERROR);
    assertThat(outErr.errAsLatin1())
        .contains(
            "ERROR: While parsing option --experimental_thread_dump_interval=1ns: Duration '1ns'"
                + " must be zero or at least 1s.");
  }

  @Test
  public void exec_overflowInterval_returnsCommandLineError(
      @TestParameter({"9223372036854776s", "200000000000000d", "99999999999999999999s"})
          String interval)
      throws Exception {
    BlazeCommandResult result =
        dispatcher.exec(
            ImmutableList.of(
                "noop",
                "--experimental_enable_thread_dump",
                "--experimental_thread_dump_interval=" + interval),
            "test",
            outErr);

    assertThat(result.getExitCode()).isEqualTo(ExitCode.COMMAND_LINE_ERROR);
    assertThat(result.getDetailedExitCode().getFailureDetail().getCommand().getCode())
        .isEqualTo(FailureDetails.Command.Code.OPTIONS_PARSE_FAILURE);
    assertThat(outErr.errAsLatin1())
        .contains("ERROR: While parsing option --experimental_thread_dump_interval=" + interval);
  }

  @Test
  public void exec_validInterval_succeeds(@TestParameter({"0", "1s"}) String interval)
      throws Exception {
    BlazeCommandResult result =
        dispatcher.exec(
            ImmutableList.of(
                "noop",
                "--experimental_enable_thread_dump",
                "--experimental_thread_dump_interval=" + interval),
            "test",
            outErr);

    assertThat(result.getExitCode()).isEqualTo(ExitCode.SUCCESS);
  }
}
