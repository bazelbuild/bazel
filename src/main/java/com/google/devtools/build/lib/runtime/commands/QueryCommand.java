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

import static com.google.devtools.build.lib.packages.Rule.ALL_LABELS;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.query2.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryUtil;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AggregateAllOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.query2.output.OutputFormatter.StreamedFormatter;
import com.google.devtools.build.lib.query2.output.QueryOptions;
import com.google.devtools.build.lib.query2.output.QueryOutputUtils;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.syntax.SkylarkSemanticsOptions;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.channels.ClosedByInterruptException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Set;

/**
 * Command line wrapper for executing a query with blaze.
 */
@Command(name = "query",
         options = { PackageCacheOptions.class,
                     SkylarkSemanticsOptions.class,
                     QueryOptions.class },
         help = "resource:query.txt",
         shortDescription = "Executes a dependency graph query.",
         allowResidue = true,
         binaryStdOut = true,
         completion = "label",
         canRunInOutputDirectory = true)
public final class QueryCommand implements BlazeCommand {

  @Override
  public void editOptions(OptionsParser optionsParser) { }

  /**
   * Exit codes:
   *   0   on successful evaluation.
   *   1   if query evaluation did not complete.
   *   2   if query parsing failed.
   *   3   if errors were reported but evaluation produced a partial result
   *        (only when --keep_going is in effect.)
   */
  @Override
  public ExitCode exec(CommandEnvironment env, OptionsProvider options) {
    BlazeRuntime runtime = env.getRuntime();
    QueryOptions queryOptions = options.getOptions(QueryOptions.class);

    try {
      env.setupPackageCache(options, runtime.getDefaultsPackageContent());
    } catch (InterruptedException e) {
      env.getReporter().handle(Event.error("query interrupted"));
      return ExitCode.INTERRUPTED;
    } catch (AbruptExitException e) {
      env.getReporter().handle(Event.error(null, "Unknown error: " + e.getMessage()));
      return e.getExitCode();
    }

    String query;
    if (!options.getResidue().isEmpty()) {
      if (!queryOptions.queryFile.isEmpty()) {
        env.getReporter()
            .handle(Event.error("Command-line query and --query_file cannot both be specified"));
        return ExitCode.COMMAND_LINE_ERROR;
      }
      query = Joiner.on(' ').join(options.getResidue());
    } else if (!queryOptions.queryFile.isEmpty()) {
      // Works for absolute or relative query file.
      Path residuePath = env.getWorkingDirectory().getRelative(queryOptions.queryFile);
      try {
        query = new String(FileSystemUtils.readContent(residuePath), StandardCharsets.UTF_8);
      } catch (IOException e) {
        env.getReporter()
            .handle(Event.error("I/O error reading from " + residuePath.getPathString()));
        return ExitCode.COMMAND_LINE_ERROR;
      }
    } else {
      env.getReporter().handle(Event.error(String.format(
          "missing query expression. Type '%s help query' for syntax and help",
          runtime.getProductName())));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    Iterable<OutputFormatter> formatters = runtime.getQueryOutputFormatters();
    OutputFormatter formatter =
        OutputFormatter.getFormatter(formatters, queryOptions.outputFormat);
    if (formatter == null) {
      env.getReporter().handle(Event.error(
          String.format("Invalid output format '%s'. Valid values are: %s",
              queryOptions.outputFormat, OutputFormatter.formatterNames(formatters))));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    Set<Setting> settings = queryOptions.toSettings();
    boolean streamResults = QueryOutputUtils.shouldStreamResults(queryOptions, formatter);
    QueryEvalResult result;
    AbstractBlazeQueryEnvironment<Target> queryEnv =
        newQueryEnvironment(
          env,
          queryOptions.keepGoing,
          !streamResults,
          queryOptions.universeScope,
          queryOptions.loadingPhaseThreads,
          settings);
    QueryExpression expr;
    try {
      expr = QueryExpression.parse(query, queryEnv);
    } catch (QueryException e) {
      env.getReporter()
          .handle(Event.error(null, "Error while parsing '" + query + "': " + e.getMessage()));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    try {
      formatter.verifyCompatible(queryEnv, expr);
    } catch (QueryException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    expr = queryEnv.transformParsedQuery(expr);

    OutputStream out = env.getReporter().getOutErr().getOutputStream();
    ThreadSafeOutputFormatterCallback<Target> callback;
    if (streamResults) {
      disableAnsiCharactersFiltering(env);
      StreamedFormatter streamedFormatter = ((StreamedFormatter) formatter);
      streamedFormatter.setOptions(
          queryOptions,
          queryOptions.aspectDeps.createResolver(env.getPackageManager(), env.getReporter()));
      callback = streamedFormatter.createStreamCallback(out, queryOptions, queryEnv);
    } else {
      callback = QueryUtil.newOrderedAggregateAllOutputFormatterCallback(queryEnv);
    }
    boolean catastrophe = true;
    try {
      result = queryEnv.evaluateQuery(expr, callback);
      catastrophe = false;
    } catch (QueryException e) {
      catastrophe = false;
      // Keep consistent with reportBuildFileError()
      env.getReporter()
          // TODO(bazel-team): this is a kludge to fix a bug observed in the wild. We should make
          // sure no null error messages ever get in.
          .handle(Event.error(e.getMessage() == null ? e.toString() : e.getMessage()));
      return ExitCode.ANALYSIS_FAILURE;
    } catch (InterruptedException e) {
      catastrophe = false;
      IOException ioException = callback.getIoException();
      if (ioException == null || ioException instanceof ClosedByInterruptException) {
        env.getReporter().handle(Event.error("query interrupted"));
        return ExitCode.INTERRUPTED;
      } else {
        env.getReporter().handle(Event.error("I/O error: " + e.getMessage()));
        return ExitCode.LOCAL_ENVIRONMENTAL_ERROR;
      }
    } catch (IOException e) {
      catastrophe = false;
      env.getReporter().handle(Event.error("I/O error: " + e.getMessage()));
      return ExitCode.LOCAL_ENVIRONMENTAL_ERROR;
    } finally {
      if (!catastrophe) {
        try {
          out.flush();
        } catch (IOException e) {
          env.getReporter().handle(
              Event.error("Failed to flush query results: " + e.getMessage()));
          return ExitCode.LOCAL_ENVIRONMENTAL_ERROR;
        }
      }
    }

    env.getEventBus().post(new NoBuildEvent(env.getCommandName(), env.getCommandStartTime()));
    if (!streamResults) {
      disableAnsiCharactersFiltering(env);
      try {
        Set<Target> targets =
            ((AggregateAllOutputFormatterCallback<Target, ?>) callback).getResult();
        QueryOutputUtils.output(
            queryOptions,
            result,
            targets,
            formatter,
            env.getReporter().getOutErr().getOutputStream(),
            queryOptions.aspectDeps.createResolver(env.getPackageManager(), env.getReporter()));
      } catch (ClosedByInterruptException | InterruptedException e) {
        env.getReporter().handle(Event.error("query interrupted"));
        return ExitCode.INTERRUPTED;
      } catch (IOException e) {
        env.getReporter().handle(Event.error("I/O error: " + e.getMessage()));
        return ExitCode.LOCAL_ENVIRONMENTAL_ERROR;
      } finally {
        try {
          out.flush();
        } catch (IOException e) {
          env.getReporter().handle(
              Event.error("Failed to flush query results: " + e.getMessage()));
          return ExitCode.LOCAL_ENVIRONMENTAL_ERROR;
        }
      }
    }

    if (result.isEmpty()) {
      env.getReporter().handle(Event.info("Empty results"));
    }

    return result.getSuccess() ? ExitCode.SUCCESS : ExitCode.PARTIAL_ANALYSIS_FAILURE;
  }

  /**
   * When Blaze is used with --color=no or not in a tty a ansi characters filter is set so that
   * we don't print fancy colors in non-supporting terminal outputs. But query output, specifically
   * the binary formatters, can print actual data that contain ansi bytes/chars. Because of that
   * we need to remove the filtering before printing any query result.
   */
  private static void disableAnsiCharactersFiltering(CommandEnvironment env) {
    env.getReporter().switchToAnsiAllowingHandler();
  }

  @VisibleForTesting // for com.google.devtools.deps.gquery.test.QueryResultTestUtil
  public static AbstractBlazeQueryEnvironment<Target> newQueryEnvironment(CommandEnvironment env,
      boolean keepGoing, boolean orderedResults, int loadingPhaseThreads,
      Set<Setting> settings) {
    return newQueryEnvironment(env, keepGoing, orderedResults, ImmutableList.<String>of(),
        loadingPhaseThreads, settings);
  }

  public static AbstractBlazeQueryEnvironment<Target> newQueryEnvironment(CommandEnvironment env,
      boolean keepGoing, boolean orderedResults, List<String> universeScope,
      int loadingPhaseThreads, Set<Setting> settings) {
    return env.getRuntime()
        .getQueryEnvironmentFactory()
        .create(
            env.getPackageManager().newTransitiveLoader(),
            env.getSkyframeExecutor(),
            env.getPackageManager(),
            env.newTargetPatternEvaluator(),
            keepGoing,
            /*strictScope=*/ true,
            orderedResults,
            universeScope,
            loadingPhaseThreads,
            /*labelFilter=*/ ALL_LABELS,
            env.getReporter(),
            settings,
            env.getRuntime().getQueryFunctions(),
            env.getPackageManager().getPackagePath(),
            /*blockUniverseEvaluationErrors=*/ false);
  }
}
