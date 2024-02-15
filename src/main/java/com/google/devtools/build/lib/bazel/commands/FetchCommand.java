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
package com.google.devtools.build.lib.bazel.commands;

import static com.google.common.primitives.Booleans.countTrue;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.analysis.NoBuildRequestFinishedEvent;
import com.google.devtools.build.lib.bazel.bzlmod.BazelFetchAllValue;
import com.google.devtools.build.lib.bazel.commands.RepositoryFetcher.RepositoryFetcherException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern.Parser;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.common.UniverseScope;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QuerySyntaxException;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.runtime.commands.QueryCommand;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.FetchCommand.Code;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue.RepositoryMappingResolutionException;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.util.EnumSet;
import java.util.List;

/** Fetches external repositories. Which is so fetch. */
@Command(
    name = FetchCommand.NAME,
    options = {
      FetchOptions.class,
      PackageOptions.class,
      KeepGoingOption.class,
      LoadingPhaseThreadsOption.class
    },
    help = "resource:fetch.txt",
    shortDescription = "Fetches external repositories that are prerequisites to the targets.",
    allowResidue = true,
    completion = "label")
public final class FetchCommand implements BlazeCommand {
  // TODO(kchodorow): this would be a great time to check for difference and invalidate the upward
  //                  transitive closure for local repositories.

  public static final String NAME = "fetch";

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    PackageOptions pkgOptions = options.getOptions(PackageOptions.class);
    if (!pkgOptions.fetch) {
      return createFailedBlazeCommandResult(
          env.getReporter(), Code.OPTIONS_INVALID, "You cannot run fetch with --nofetch");
    }
    FetchOptions fetchOptions = options.getOptions(FetchOptions.class);
    int optionsCount =
        countTrue(
            fetchOptions.all,
            fetchOptions.configure,
            !fetchOptions.repos.isEmpty(),
            !options.getResidue().isEmpty());
    if (optionsCount > 1) {
      return createFailedBlazeCommandResult(
          env.getReporter(),
          Code.OPTIONS_INVALID,
          "Only one fetch option should be provided for fetch command.");
    }
    LoadingPhaseThreadsOption threadsOption = options.getOptions(LoadingPhaseThreadsOption.class);

    env.getEventBus()
        .post(
            new NoBuildEvent(
                env.getCommandName(),
                env.getCommandStartTime(),
                /* separateFinishedEvent= */ true,
                /* showProgress= */ true,
                /* id= */ null));
    BlazeCommandResult result;
    if (fetchOptions.force) {
      // Using commandId as the value -instead of true/false- to make sure to invalidate skyframe
      // and to actually force fetch each time
      env.getSkyframeExecutor()
          .injectExtraPrecomputedValues(
              ImmutableList.of(
                  PrecomputedValue.injected(
                      RepositoryDelegatorFunction.FORCE_FETCH, env.getCommandId().toString())));
    }
    try {
      env.syncPackageLoading(options);
      if (fetchOptions.all || fetchOptions.configure) {
      result = fetchAll(env, options, threadsOption, fetchOptions.configure);
      } else if (!fetchOptions.repos.isEmpty()) {
        result = fetchRepos(env, threadsOption, fetchOptions.repos);
      } else {
        result = fetchTarget(env, options, threadsOption);
      }
    } catch (AbruptExitException e) {
      return createFailedBlazeCommandResult(
          env.getReporter(), e.getMessage(), e.getDetailedExitCode());
    } catch (InterruptedException e) {
      return createFailedBlazeCommandResult(
          env.getReporter(), "Fetch interrupted: " + e.getMessage());
    }
    env.getEventBus()
        .post(
            new NoBuildRequestFinishedEvent(
                result.getExitCode(), env.getRuntime().getClock().currentTimeMillis()));
    return result;
  }

  private BlazeCommandResult fetchAll(
      CommandEnvironment env,
      OptionsParsingResult options,
      LoadingPhaseThreadsOption threadsOption,
      boolean configureEnabled)
      throws InterruptedException {
    if (!options.getOptions(BuildLanguageOptions.class).enableBzlmod) {
      return createFailedBlazeCommandResult(
          env.getReporter(),
          "Bzlmod has to be enabled for fetch --all to work, run with --enable_bzlmod");
    }

    SkyframeExecutor skyframeExecutor = env.getSkyframeExecutor();
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setParallelism(threadsOption.threads)
            .setEventHandler(env.getReporter())
            .build();

    EvaluationResult<SkyValue> evaluationResult =
        skyframeExecutor.prepareAndGet(
            ImmutableSet.of(BazelFetchAllValue.key(configureEnabled)), evaluationContext);
    if (evaluationResult.hasError()) {
      Exception e = evaluationResult.getError().getException();
      return createFailedBlazeCommandResult(
          env.getReporter(),
          e != null ? e.getMessage() : "Unexpected error during fetching all external deps.");
    }
    return BlazeCommandResult.success();
  }

  private BlazeCommandResult fetchRepos(
      CommandEnvironment env, LoadingPhaseThreadsOption threadsOption, List<String> repos)
      throws InterruptedException {
    try {
      ImmutableMap<RepositoryName, RepositoryDirectoryValue> repositoryNamesAndValues =
          RepositoryFetcher.fetchRepos(repos, env, threadsOption);
      String notFoundRepos =
          repositoryNamesAndValues.values().stream()
              .filter(value -> !value.repositoryExists())
              .map(value -> value.getErrorMsg())
              .collect(joining("; "));
      if (!notFoundRepos.isEmpty()) {
        return createFailedBlazeCommandResult(
            env.getReporter(), "Fetching repos failed with errors: " + notFoundRepos);
      }
      return BlazeCommandResult.success();
    } catch (RepositoryMappingResolutionException e) {
      return createFailedBlazeCommandResult(
          env.getReporter(), "Invalid repo name: " + e.getMessage(), e.getDetailedExitCode());
    } catch (RepositoryFetcherException e) {
      return createFailedBlazeCommandResult(env.getReporter(), e.getMessage());
    }
  }

  private BlazeCommandResult fetchTarget(
      CommandEnvironment env, OptionsParsingResult options, LoadingPhaseThreadsOption threadsOption)
      throws InterruptedException {
    if (options.getResidue().isEmpty()) {
      return createFailedBlazeCommandResult(
          env.getReporter(),
          Code.EXPRESSION_MISSING,
          String.format(
              "missing fetch expression. Type '%s help fetch' for syntax and help",
              env.getRuntime().getProductName()));
    }

    boolean keepGoing = options.getOptions(KeepGoingOption.class).keepGoing;
    TargetPattern.Parser mainRepoTargetParser;
    try {
      RepositoryMapping repoMapping =
          env.getSkyframeExecutor()
              .getMainRepoMapping(keepGoing, threadsOption.threads, env.getReporter());

      mainRepoTargetParser =
          new Parser(env.getRelativeWorkingDirectory(), RepositoryName.MAIN, repoMapping);
    } catch (RepositoryMappingResolutionException e) {
      return createFailedBlazeCommandResult(
          env.getReporter(), e.getMessage(), e.getDetailedExitCode());
    }

    // Querying for all of the dependencies of the targets has the side-effect of populating the
    // Skyframe graph for external targets, which requires downloading them. The JDK is required to
    // build everything but isn't counted as a dep in the build graph so we add it manually.
    ImmutableList.Builder<String> labelsToLoad =
        new ImmutableList.Builder<String>().addAll(options.getResidue());

    String query = Joiner.on(" union ").join(labelsToLoad.build());
    query = "deps(" + query + ")";

    AbstractBlazeQueryEnvironment<Target> queryEnv =
        QueryCommand.newQueryEnvironment(
            env,
            keepGoing,
            false,
            UniverseScope.EMPTY,
            threadsOption.threads,
            EnumSet.noneOf(Setting.class),
            /* useGraphlessQuery= */ true,
            mainRepoTargetParser,
            LabelPrinter.legacy());

    // 1. Parse query:
    QueryExpression expr;
    try {
      expr = QueryExpression.parse(query, queryEnv);
    } catch (QuerySyntaxException e) {
      return createFailedBlazeCommandResult(
          env.getReporter(),
          Code.QUERY_PARSE_ERROR,
          String.format(
              "Error while parsing '%s': %s", QueryExpression.truncate(query), e.getMessage()));
    }

    env.getReporter()
        .post(
            new NoBuildEvent(
                env.getCommandName(),
                env.getCommandStartTime(),
                true,
                true,
                env.getCommandId().toString()));

    // 2. Evaluate expression:
    QueryEvalResult queryEvalResult = null;
    try {
      queryEvalResult =
          queryEnv.evaluateQuery(
              expr,
              new ThreadSafeOutputFormatterCallback<Target>() {
                @Override
                public void processOutput(Iterable<Target> partialResult) {
                  // Throw away the result.
                }
              });
    } catch (InterruptedException e) {
      env.getReporter()
          .post(
              new NoBuildRequestFinishedEvent(
                  ExitCode.COMMAND_LINE_ERROR, env.getRuntime().getClock().currentTimeMillis()));
      return BlazeCommandResult.detailedExitCode(
          InterruptedFailureDetails.detailedExitCode(e.getMessage()));
    } catch (QueryException e) {
      // Keep consistent with reportBuildFileError()
      env.getReporter()
          .post(
              new NoBuildRequestFinishedEvent(
                  ExitCode.COMMAND_LINE_ERROR, env.getRuntime().getClock().currentTimeMillis()));
      return createFailedBlazeCommandResult(
          env.getReporter(), Code.QUERY_PARSE_ERROR, e.getMessage());
    } catch (IOException e) {
      // Should be impossible since our OutputFormatterCallback doesn't throw IOException.
      throw new IllegalStateException(e);
    }

    if (queryEvalResult.getSuccess()) {
      env.getReporter().handle(Event.info("All external dependencies fetched successfully."));
    }
    env.getReporter()
        .post(
            new NoBuildRequestFinishedEvent(
                queryEvalResult.getSuccess() ? ExitCode.SUCCESS : ExitCode.COMMAND_LINE_ERROR,
                env.getRuntime().getClock().currentTimeMillis()));
    return queryEvalResult.getSuccess()
        ? BlazeCommandResult.success()
        : createFailedBlazeCommandResult(
            env.getReporter(),
            Code.QUERY_EVALUATION_ERROR,
            String.format(
                "Evaluation of query \"%s\" failed but --keep_going specified, ignoring errors",
                expr));
  }

  private static BlazeCommandResult createFailedBlazeCommandResult(
      Reporter reporter, Code fetchCommandCode, String message) {
    return createFailedBlazeCommandResult(
        reporter,
        message,
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setFetchCommand(
                    FailureDetails.FetchCommand.newBuilder().setCode(fetchCommandCode).build())
                .build()));
  }

  private static BlazeCommandResult createFailedBlazeCommandResult(
      Reporter reporter, String errorMessage) {
    return createFailedBlazeCommandResult(
        reporter, errorMessage, InterruptedFailureDetails.detailedExitCode(errorMessage));
  }

  private static BlazeCommandResult createFailedBlazeCommandResult(
      Reporter reporter, String message, DetailedExitCode exitCode) {
    reporter.handle(Event.error(message));
    return BlazeCommandResult.detailedExitCode(exitCode);
  }
}
