// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.blaze.commands;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.blaze.AggregatingTestListener;
import com.google.devtools.build.lib.blaze.BlazeCommand;
import com.google.devtools.build.lib.blaze.BlazeCommandEventHandler;
import com.google.devtools.build.lib.blaze.BlazeRuntime;
import com.google.devtools.build.lib.blaze.Command;
import com.google.devtools.build.lib.blaze.TerminalTestResultNotifier;
import com.google.devtools.build.lib.blaze.TerminalTestResultNotifier.TestSummaryOptions;
import com.google.devtools.build.lib.blaze.TestResultAnalyzer;
import com.google.devtools.build.lib.blaze.TestResultNotifier;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.TestStrategy;
import com.google.devtools.build.lib.exec.TestStrategy.TestOutputFormat;
import com.google.devtools.build.lib.util.ExitCausingException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.AnsiTerminalPrinter;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;

import java.util.Collection;

/**
 * Handles the 'test' command on the Blaze command line.
 */
@Command(name = "test",
         builds = true,
         inherits = { BuildCommand.class },
         options = { TestSummaryOptions.class },
         shortDescription = "Builds and runs the specified test targets.",
         help = "resource:test.txt",
         allowResidue = true)
public class TestCommand implements BlazeCommand {
  private AnsiTerminalPrinter printer;

  @Override
  public void editOptions(BlazeRuntime runtime, OptionsParser optionsParser)
      throws ExitCausingException {
    TestOutputFormat testOutput = optionsParser.getOptions(ExecutionOptions.class).testOutput;

    if (testOutput == TestStrategy.TestOutputFormat.STREAMED) {
      runtime.getReporter().warn(null,
          "Streamed test output requested so all tests will be run locally, without sharding, " +
           "one at a time");
      try {
        optionsParser.parse(OptionPriority.SOFTWARE_REQUIREMENT,
            "streamed output requires locally run tests, without sharding",
            ImmutableList.of("--test_sharding_strategy=disabled", "--test_strategy=exclusive"));
      } catch (OptionsParsingException e) {
        throw new IllegalStateException("Known options failed to parse", e);
      }
    }
  }

  @Override
  public ExitCode exec(BlazeRuntime runtime, OptionsProvider options, OutErr outErr) {
    TestResultAnalyzer resultAnalyzer = new TestResultAnalyzer(
        runtime.getExecRoot(),
        options.getOptions(TestSummaryOptions.class),
        options.getOptions(ExecutionOptions.class),
        runtime.getEventBus());

    printer = new AnsiTerminalPrinter(outErr.getOutputStream(),
        options.getOptions(BlazeCommandEventHandler.Options.class).useColor());

    // Initialize test handler.
    AggregatingTestListener testListener = new AggregatingTestListener(
        resultAnalyzer, runtime.getEventBus(), runtime.getReporter());

    runtime.getEventBus().register(testListener);
    return doTest(runtime, options, testListener, outErr);
  }

  private ExitCode doTest(BlazeRuntime runtime,
                          OptionsProvider options,
                          AggregatingTestListener testListener,
                          OutErr outErr) {
    // Run simultaneous build and test.
    BuildRequest request = BuildRequest.create(
        getClass().getAnnotation(Command.class).name(), options,
        runtime.getStartupOptionsProvider(), options.getResidue(),
        outErr, runtime.getCommandId(), runtime.getCommandStartTime());
    if (request.getBuildOptions().compileOnly) {
      String message =  "The '" + getClass().getAnnotation(Command.class).name() +
                        "' command is incompatible with the --compile_only option";
      runtime.getReporter().error(null, message);
      return ExitCode.COMMAND_LINE_ERROR;
    }
    request.setRunTestsDuringBuild();

    BuildResult buildResult = runtime.getBuildTool().processRequest(request, null);

    Collection<ConfiguredTarget> testTargets = buildResult.getTestTargets();
    Collection<ConfiguredTarget> successfulTargets = buildResult.getSuccessfulTargets();
    // TODO(bazel-team): don't handle isEmpty here or fix up a bunch of tests
    if (successfulTargets == null) {
      // This can happen if there were errors in the target parsing or loading phase
      // (original exitcode=BUILD_FAILURE) or if there weren't but --noanalyze was given
      // (original exitcode=SUCCESS).
      runtime.getReporter().error(null, "Couldn't start the build. Unable to run tests");
      return buildResult.getSuccess() ? ExitCode.PARSING_FAILURE : buildResult.getExitCondition();
    }
    // TODO(bazel-team): the check above shadows NO_TESTS_FOUND, but switching the conditions breaks
    // more tests
    if (testTargets.isEmpty()) {
      runtime.getReporter().error(
          null, "No test targets were found, yet testing was requested");
      return buildResult.getSuccess() ? ExitCode.NO_TESTS_FOUND : buildResult.getExitCondition();
    }

    boolean buildSuccess = buildResult.getSuccess();
    boolean testSuccess = analyzeTestResults(testTargets, successfulTargets, testListener, options);

    if (testSuccess && !buildSuccess) {
      // If all tests run successfully, test summary should include warning if
      // there were build errors not associated with the test targets.
      printer.printLn(AnsiTerminalPrinter.Mode.ERROR
          + "One or more non-test targets failed to build.\n"
          + AnsiTerminalPrinter.Mode.DEFAULT);
    }

    return buildSuccess ?
           (testSuccess ? ExitCode.SUCCESS : ExitCode.TESTS_FAILED)
           : buildResult.getExitCondition();
  }

  /**
   * Analyzes test results and prints summary information.
   * Returns true if and only if all tests were successful.
   */
  private boolean analyzeTestResults(Collection<ConfiguredTarget> testTargets,
                                     Collection<ConfiguredTarget> successfulTargets,
                                     AggregatingTestListener listener,
                                     OptionsProvider options) {
    TestResultNotifier notifier = new TerminalTestResultNotifier(printer, options);
    return listener.getAnalyzer().differentialAnalyzeAndReport(
        testTargets, successfulTargets, listener, notifier);
  }
}
