// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.BuildTool;
import com.google.devtools.build.lib.buildtool.InstrumentationFilterSupport;
import com.google.devtools.build.lib.buildtool.buildevent.TestingCompleteEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.TestStrategy;
import com.google.devtools.build.lib.exec.TestStrategy.TestOutputFormat;
import com.google.devtools.build.lib.runtime.AggregatingTestListener;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandEventHandler;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.TerminalTestResultNotifier;
import com.google.devtools.build.lib.runtime.TerminalTestResultNotifier.TestSummaryOptions;
import com.google.devtools.build.lib.runtime.TestResultAnalyzer;
import com.google.devtools.build.lib.runtime.TestResultNotifier;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.AnsiTerminalPrinter;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;
import java.util.Collection;
import java.util.List;

/**
 * Handles the 'test' command on the Blaze command line.
 */
@Command(name = "test",
         builds = true,
         inherits = { BuildCommand.class },
         options = { TestSummaryOptions.class },
         shortDescription = "Builds and runs the specified test targets.",
         help = "resource:test.txt",
         completion = "label-test",
         allowResidue = true)
public class TestCommand implements BlazeCommand {
  private AnsiTerminalPrinter printer;

  /** Returns the name of the command to ask the project file for. */
  // TODO(hdm): move into BlazeRuntime?  It feels odd to duplicate the annotation here.
  protected String commandName() {
    return "test";
  }

  @Override
  public void editOptions(OptionsParser optionsParser) {
    TestOutputFormat testOutput = optionsParser.getOptions(ExecutionOptions.class).testOutput;
    try {
      if (testOutput == TestStrategy.TestOutputFormat.STREAMED) {
        optionsParser.parse(OptionPriority.SOFTWARE_REQUIREMENT,
            "streamed output requires locally run tests, without sharding",
            ImmutableList.of("--test_sharding_strategy=disabled", "--test_strategy=exclusive"));
      }
    } catch (OptionsParsingException e) {
      throw new IllegalStateException("Known options failed to parse", e);
    }
  }

  @Override
  public ExitCode exec(CommandEnvironment env, OptionsProvider options) {
    TestOutputFormat testOutput = options.getOptions(ExecutionOptions.class).testOutput;
    if (testOutput == TestStrategy.TestOutputFormat.STREAMED) {
      env.getReporter().handle(Event.warn(
          "Streamed test output requested. All tests will be run locally, without sharding, "
          + "one at a time"));
    }

    TestResultAnalyzer resultAnalyzer = new TestResultAnalyzer(
        options.getOptions(TestSummaryOptions.class),
        options.getOptions(ExecutionOptions.class),
        env.getEventBus());

    printer = new AnsiTerminalPrinter(env.getReporter().getOutErr().getOutputStream(),
        options.getOptions(BlazeCommandEventHandler.Options.class).useColor());

    // Initialize test handler.
    AggregatingTestListener testListener = new AggregatingTestListener(
        resultAnalyzer, env.getEventBus(), env.getReporter());

    env.getEventBus().register(testListener);
    return doTest(env, options, testListener);
  }

  private ExitCode doTest(CommandEnvironment env,
      OptionsProvider options,
      AggregatingTestListener testListener) {
    BlazeRuntime runtime = env.getRuntime();
    // Run simultaneous build and test.
    List<String> targets = ProjectFileSupport.getTargets(runtime.getProjectFileProvider(), options);
    BuildRequest request = BuildRequest.create(
        getClass().getAnnotation(Command.class).name(), options,
        runtime.getStartupOptionsProvider(), targets,
        env.getReporter().getOutErr(), env.getCommandId(), env.getCommandStartTime());
    request.setRunTests();
    if (options.getOptions(BuildConfiguration.Options.class).collectCodeCoverage
        && !options.containsExplicitOption(
            InstrumentationFilterSupport.INSTRUMENTATION_FILTER_FLAG)) {
      request.setNeedsInstrumentationFilter(true);
    }

    BuildResult buildResult = new BuildTool(env).processRequest(request, null);

    Collection<ConfiguredTarget> testTargets = buildResult.getTestTargets();
    // TODO(bazel-team): don't handle isEmpty here or fix up a bunch of tests
    if (buildResult.getSuccessfulTargets() == null) {
      // This can happen if there were errors in the target parsing or loading phase
      // (original exitcode=BUILD_FAILURE) or if there weren't but --noanalyze was given
      // (original exitcode=SUCCESS).
      env.getReporter().handle(Event.error("Couldn't start the build. Unable to run tests"));
      return buildResult.getSuccess() ? ExitCode.PARSING_FAILURE : buildResult.getExitCondition();
    }
    // TODO(bazel-team): the check above shadows NO_TESTS_FOUND, but switching the conditions breaks
    // more tests
    if (testTargets.isEmpty()) {
      env.getReporter().handle(Event.error(
          null, "No test targets were found, yet testing was requested"));
      return buildResult.getSuccess() ? ExitCode.NO_TESTS_FOUND : buildResult.getExitCondition();
    }

    boolean buildSuccess = buildResult.getSuccess();
    boolean testSuccess = analyzeTestResults(testTargets, testListener, options);

    if (testSuccess && !buildSuccess) {
      // If all tests run successfully, test summary should include warning if
      // there were build errors not associated with the test targets.
      printer.printLn(AnsiTerminalPrinter.Mode.ERROR
          + "All tests passed but there were other errors during the build.\n"
          + AnsiTerminalPrinter.Mode.DEFAULT);
    }

    ExitCode exitCode = buildSuccess
        ? (testSuccess ? ExitCode.SUCCESS : ExitCode.TESTS_FAILED)
        : buildResult.getExitCondition();
    env.getEventBus().post(new TestingCompleteEvent(exitCode, buildResult.getStopTime()));
    return exitCode;
  }

  /**
   * Analyzes test results and prints summary information.
   * Returns true if and only if all tests were successful.
   */
  private boolean analyzeTestResults(Collection<ConfiguredTarget> testTargets,
                                     AggregatingTestListener listener,
                                     OptionsProvider options) {
    TestResultNotifier notifier = new TerminalTestResultNotifier(printer, options);
    return listener.getAnalyzer().differentialAnalyzeAndReport(
        testTargets, listener, notifier);
  }
}
