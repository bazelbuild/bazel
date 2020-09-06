// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.analysis.NoBuildRequestFinishedEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.common.UniverseScope;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.query.output.OutputFormatter;
import com.google.devtools.build.lib.query2.query.output.OutputFormatters;
import com.google.devtools.build.lib.query2.query.output.QueryOptions;
import com.google.devtools.build.lib.query2.query.output.QueryOutputUtils;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.runtime.QueryRuntimeHelper;
import com.google.devtools.build.lib.runtime.QueryRuntimeHelper.QueryRuntimeHelperException;
import com.google.devtools.build.lib.runtime.TargetProviderForQueryEnvironment;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted;
import com.google.devtools.build.lib.server.FailureDetails.Query;
import com.google.devtools.build.lib.skyframe.LoadingPhaseStartedEvent;
import com.google.devtools.build.lib.skyframe.PackageProgressReceiver;
import com.google.devtools.build.lib.skyframe.SkyframeExecutorWrappingWalkableGraph;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Either;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.devtools.common.options.TriState;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Set;
import java.util.function.Function;

/**
 * Common methods and utils to set up Blaze Runtime environments for {@link BlazeCommand} which
 * requires {@link QueryEnvironment}
 */
public abstract class QueryEnvironmentBasedCommand implements BlazeCommand {

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
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    env.getEventBus()
        .post(
            new NoBuildEvent(
                env.getCommandName(),
                env.getCommandStartTime(),
                /* separateFinishedEvent= */ true,
                /* showProgress= */ true,
                /* id= */ null));
    BlazeCommandResult result = execInternal(env, options);
    env.getEventBus()
        .post(
            new NoBuildRequestFinishedEvent(
                result.getExitCode(), env.getRuntime().getClock().currentTimeMillis()));
    return result;
  }

  private BlazeCommandResult execInternal(CommandEnvironment env, OptionsParsingResult options) {
    BlazeRuntime runtime = env.getRuntime();
    QueryOptions queryOptions = options.getOptions(QueryOptions.class);

    try {
      env.syncPackageLoading(options);
    } catch (InterruptedException e) {
      return reportAndCreateInterruptResult(
          env, "query interrupted", Interrupted.Code.PACKAGE_LOADING_SYNC);
    } catch (AbruptExitException e) {
      env.getReporter().handle(Event.error(null, "Unknown error: " + e.getMessage()));
      return BlazeCommandResult.detailedExitCode(e.getDetailedExitCode());
    }

    String query;
    if (!options.getResidue().isEmpty()) {
      if (!queryOptions.queryFile.isEmpty()) {
        return reportAndCreateFailureResult(
            env,
            "Command-line query and --query_file cannot both be specified",
            Query.Code.QUERY_FILE_WITH_COMMAND_LINE_EXPRESSION);
      }
      query = Joiner.on(' ').join(options.getResidue());
    } else if (!queryOptions.queryFile.isEmpty()) {
      // Works for absolute or relative query file.
      Path residuePath = env.getWorkingDirectory().getRelative(queryOptions.queryFile);
      try {
        query = new String(FileSystemUtils.readContent(residuePath), StandardCharsets.UTF_8);
      } catch (IOException e) {
        return reportAndCreateFailureResult(
            env,
            "I/O error reading from " + residuePath.getPathString(),
            Query.Code.QUERY_FILE_READ_FAILURE);
      }
    } else {
      return reportAndCreateFailureResult(
          env,
          String.format(
              "missing query expression. Type '%s help query' for syntax and help",
              runtime.getProductName()),
          Query.Code.COMMAND_LINE_EXPRESSION_MISSING);
    }

    Iterable<OutputFormatter> formatters = runtime.getQueryOutputFormatters();
    OutputFormatter formatter =
        OutputFormatters.getFormatter(formatters, queryOptions.outputFormat);
    if (formatter == null) {
      return reportAndCreateFailureResult(
          env,
          String.format(
              "Invalid output format '%s'. Valid values are: %s",
              queryOptions.outputFormat, OutputFormatters.formatterNames(formatters)),
          Query.Code.OUTPUT_FORMAT_INVALID);
    }

    Set<Setting> settings = queryOptions.toSettings();
    boolean streamResults = QueryOutputUtils.shouldStreamResults(queryOptions, formatter);
    boolean useGraphlessQuery =
        queryOptions.useGraphlessQuery == TriState.YES
            || (queryOptions.useGraphlessQuery == TriState.AUTO && streamResults);
    if (useGraphlessQuery && !streamResults) {
      return reportAndCreateFailureResult(
          env,
          String.format(
              "--experimental_graphless_query requires --order_output=no and an --output"
                  + " option that supports streaming; valid values are: %s",
              OutputFormatters.streamingFormatterNames(formatters)),
          Query.Code.GRAPHLESS_PREREQ_UNMET);
    }

    try (QueryRuntimeHelper queryRuntimeHelper =
        env.getRuntime().getQueryRuntimeHelperFactory().create(env)) {
      Either<BlazeCommandResult, QueryEvalResult> result;
      try (AbstractBlazeQueryEnvironment<Target> queryEnv =
          newQueryEnvironment(
              env,
              options.getOptions(KeepGoingOption.class).keepGoing,
              !streamResults,
              getUniverseScope(queryOptions),
              options.getOptions(LoadingPhaseThreadsOption.class).threads,
              settings,
              useGraphlessQuery)) {
        result =
            doQuery(
                query, env, queryOptions, streamResults, formatter, queryEnv, queryRuntimeHelper);
      }
      return result.map(
          Function.identity(),
          queryEvalResult -> {
            if (queryEvalResult.isEmpty()) {
              env.getReporter().handle(Event.info("Empty results"));
            }
            try {
              queryRuntimeHelper.afterQueryOutputIsWritten();
            } catch (QueryRuntimeHelperException e) {
              env.getReporter().handle(Event.error(e.getMessage()));
              return BlazeCommandResult.detailedExitCode(DetailedExitCode.of(e.getFailureDetail()));
            } catch (InterruptedException e) {
              return reportAndCreateInterruptResult(
                  env, "query interrupted", Interrupted.Code.AFTER_QUERY);
            }
            if (queryEvalResult.getSuccess()) {
              return BlazeCommandResult.success();
            }
            // The numerical exit code expected by query users in this case is always 3
            // (corresponding to ExitCode.PARTIAL_ANALYSIS_FAILURE), which is why the command
            // result returned here overrides any numerical code associated with the
            // detailedExitCode in the eval result.
            return BlazeCommandResult.detailedExitCode(
                DetailedExitCode.of(
                    ExitCode.PARTIAL_ANALYSIS_FAILURE,
                    queryEvalResult.getDetailedExitCode().getFailureDetail()));
          });
    } catch (QueryRuntimeHelperException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return BlazeCommandResult.detailedExitCode(DetailedExitCode.of(e.getFailureDetail()));
    }
  }

  private static UniverseScope getUniverseScope(QueryOptions queryOptions) {
    if (!queryOptions.universeScope.isEmpty()) {
      return UniverseScope.fromUniverseScopeList(ImmutableList.copyOf(queryOptions.universeScope));
    }
    return queryOptions.inferUniverseScope
        ? UniverseScope.INFER_FROM_QUERY_EXPRESSION
        : UniverseScope.EMPTY;
  }

  protected abstract Either<BlazeCommandResult, QueryEvalResult> doQuery(
      String query,
      CommandEnvironment env,
      QueryOptions queryOptions,
      boolean streamResults,
      OutputFormatter formatter,
      AbstractBlazeQueryEnvironment<Target> queryEnv,
      QueryRuntimeHelper queryRuntimeHelper);

  public static AbstractBlazeQueryEnvironment<Target> newQueryEnvironment(
      CommandEnvironment env,
      boolean keepGoing,
      boolean orderedResults,
      UniverseScope universeScope,
      int loadingPhaseThreads,
      Set<Setting> settings,
      boolean useGraphlessQuery) {

    WalkableGraph walkableGraph =
        SkyframeExecutorWrappingWalkableGraph.of(env.getSkyframeExecutor());

    TargetProviderForQueryEnvironment targetProviderForQueryEnvironment =
        new TargetProviderForQueryEnvironment(walkableGraph, env.getPackageManager());

    PackageProgressReceiver progressReceiver =
        env.getSkyframeExecutor().getPackageProgressReceiver();
    if (progressReceiver != null) {
      progressReceiver.reset();
      env.getReporter().post(new LoadingPhaseStartedEvent(progressReceiver));
    }

    return env.getRuntime()
        .getQueryEnvironmentFactory()
        .create(
            env.getPackageManager().newTransitiveLoader(),
            env.getSkyframeExecutor(),
            targetProviderForQueryEnvironment,
            env.getPackageManager(),
            env.getPackageManager().newTargetPatternPreloader(),
            env.getRelativeWorkingDirectory(),
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
            /*blockUniverseEvaluationErrors=*/ false,
            useGraphlessQuery);
  }

  private static BlazeCommandResult reportAndCreateInterruptResult(
      CommandEnvironment env, String message, Interrupted.Code detailedCode) {
    env.getReporter().handle(Event.error(message));
    return BlazeCommandResult.detailedExitCode(
        InterruptedFailureDetails.detailedExitCode(message, detailedCode));
  }

  private static BlazeCommandResult reportAndCreateFailureResult(
      CommandEnvironment env, String message, Query.Code detailedCode) {
    env.getReporter().handle(Event.error(message));
    return createFailureResult(message, detailedCode);
  }

  private static BlazeCommandResult createFailureResult(String message, Query.Code detailedCode) {
    return BlazeCommandResult.detailedExitCode(
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setQuery(Query.newBuilder().setCode(detailedCode))
                .build()));
  }
}
