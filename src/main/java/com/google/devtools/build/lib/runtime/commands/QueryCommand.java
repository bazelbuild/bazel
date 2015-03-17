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
package com.google.devtools.build.lib.runtime.commands;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.query2.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.query2.output.QueryOptions;
import com.google.devtools.build.lib.query2.output.QueryOutputUtils;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.channels.ClosedByInterruptException;
import java.util.List;
import java.util.Set;

/**
 * Command line wrapper for executing a query with blaze.
 */
@Command(name = "query",
         options = { PackageCacheOptions.class,
                     QueryOptions.class },
         help = "resource:query.txt",
         shortDescription = "Executes a dependency graph query.",
         allowResidue = true,
         binaryStdOut = true,
         canRunInOutputDirectory = true)
public final class QueryCommand implements BlazeCommand {

  @Override
  public void editOptions(BlazeRuntime runtime, OptionsParser optionsParser) { }

  /**
   * Exit codes:
   *   0   on successful evaluation.
   *   1   if query evaluation did not complete.
   *   2   if query parsing failed.
   *   3   if errors were reported but evaluation produced a partial result
   *        (only when --keep_going is in effect.)
   */
  @Override
  public ExitCode exec(BlazeRuntime runtime, OptionsProvider options) {
    QueryOptions queryOptions = options.getOptions(QueryOptions.class);

    try {
      runtime.setupPackageCache(
          options.getOptions(PackageCacheOptions.class),
          runtime.getDefaultsPackageContent());
    } catch (InterruptedException e) {
      runtime.getReporter().handle(Event.error("query interrupted"));
      return ExitCode.INTERRUPTED;
    } catch (AbruptExitException e) {
      runtime.getReporter().handle(Event.error(null, "Unknown error: " + e.getMessage()));
      return e.getExitCode();
    }

    if (options.getResidue().isEmpty()) {
      runtime.getReporter().handle(Event.error(String.format(
          "missing query expression. Type '%s help query' for syntax and help",
          Constants.PRODUCT_NAME)));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    Iterable<OutputFormatter> formatters = runtime.getQueryOutputFormatters();
    OutputFormatter formatter =
        OutputFormatter.getFormatter(formatters, queryOptions.outputFormat);
    if (formatter == null) {
      runtime.getReporter().handle(Event.error(
          String.format("Invalid output format '%s'. Valid values are: %s",
              queryOptions.outputFormat, OutputFormatter.formatterNames(formatters))));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    String query = Joiner.on(' ').join(options.getResidue());

    Set<Setting> settings = queryOptions.toSettings();
    AbstractBlazeQueryEnvironment<Target> env = newQueryEnvironment(
        runtime,
        queryOptions.keepGoing,
        QueryOutputUtils.orderResults(queryOptions, formatter),
        queryOptions.universeScope, queryOptions.loadingPhaseThreads,
        settings);

    // 1. Parse query:
    QueryExpression expr;
    try {
      expr = QueryExpression.parse(query, env);
    } catch (QueryException e) {
      runtime.getReporter().handle(Event.error(
          null, "Error while parsing '" + query + "': " + e.getMessage()));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    // 2. Evaluate expression:
    QueryEvalResult<Target> result;
    try {
      result = env.evaluateQuery(expr);
    } catch (QueryException e) {
      // Keep consistent with reportBuildFileError()
      runtime.getReporter().handle(Event.error(e.getMessage()));
      return ExitCode.ANALYSIS_FAILURE;
    }

    // 3. Output results:
    PrintStream output = new PrintStream(runtime.getReporter().getOutErr().getOutputStream());
    try {
      QueryOutputUtils.output(queryOptions, result, formatter, output);
    } catch (ClosedByInterruptException e) {
      runtime.getReporter().handle(Event.error("query interrupted"));
      return ExitCode.INTERRUPTED;
    } catch (IOException e) {
      runtime.getReporter().handle(Event.error("I/O error: " + e.getMessage()));
      return ExitCode.LOCAL_ENVIRONMENTAL_ERROR;
    } finally {
      output.flush();
    }
    if (result.getResultSet().isEmpty()) {
      runtime.getReporter().handle(Event.info("Empty results"));
    }

    return result.getSuccess() ? ExitCode.SUCCESS : ExitCode.PARTIAL_ANALYSIS_FAILURE;
  }

  @VisibleForTesting // for com.google.devtools.deps.gquery.test.QueryResultTestUtil
  public static AbstractBlazeQueryEnvironment<Target> newQueryEnvironment(BlazeRuntime runtime,
      boolean keepGoing, boolean orderedResults, int loadingPhaseThreads,
      Set<Setting> settings) {
    return newQueryEnvironment(runtime, keepGoing, orderedResults, ImmutableList.<String>of(),
        loadingPhaseThreads, settings);
  }

  public static AbstractBlazeQueryEnvironment<Target> newQueryEnvironment(BlazeRuntime runtime,
      boolean keepGoing, boolean orderedResults, List<String> universeScope,
      int loadingPhaseThreads,
      Set<Setting> settings) {
    ImmutableList.Builder<QueryFunction> functions = ImmutableList.builder();
    for (BlazeModule module : runtime.getBlazeModules()) {
      functions.addAll(module.getQueryFunctions());
    }
    return AbstractBlazeQueryEnvironment.newQueryEnvironment(
        runtime.getPackageManager().newTransitiveLoader(),
        runtime.getSkyframeExecutor(),
        runtime.getPackageManager(),
        runtime.getTargetPatternEvaluator(),
        keepGoing, orderedResults, universeScope, loadingPhaseThreads, runtime.getReporter(),
        settings,
        functions.build(),
        runtime.getPackageManager().getPackagePath());
  }
}
