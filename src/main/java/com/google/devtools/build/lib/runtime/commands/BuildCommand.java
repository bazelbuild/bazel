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

import static com.google.devtools.build.lib.runtime.Command.BuildPhase.EXECUTES;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.AnalysisOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildtool.BuildCqueryProcessor;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.buildtool.BuildTool;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern.Parser;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.common.UniverseScopeOptions;
import com.google.devtools.build.lib.query2.cquery.ConfiguredTargetQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.query2.engine.QuerySyntaxException;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery;
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue.RepositoryMappingResolutionException;
import com.google.devtools.build.lib.skyframe.SkyfocusOptions;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCachingOptions;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * Handles the 'build' command on the Blaze command line, including targets named by arguments
 * passed to Blaze.
 */
@Command(
    name = "build",
    buildPhase = EXECUTES,
    options = {
      BuildRequestOptions.class,
      BuildCqueryOptions.class,
      UniverseScopeOptions.class,
      ExecutionOptions.class,
      LocalExecutionOptions.class,
      PackageOptions.class,
      AnalysisOptions.class,
      LoadingOptions.class,
      KeepGoingOption.class,
      LoadingPhaseThreadsOption.class,
      BuildEventProtocolOptions.class,
      SkyfocusOptions.class,
      RemoteAnalysisCachingOptions.class,
    },
    usesConfigurationOptions = true,
    shortDescription = "Builds the specified targets.",
    allowResidue = true,
    completion = "label",
    help = "resource:build.txt")
public final class BuildCommand implements BlazeCommand {

  @Override
  public void editOptions(OptionsParser optionsParser) {
    BuildCqueryOptions cqueryOptions = optionsParser.getOptions(BuildCqueryOptions.class);
    UniverseScopeOptions universeScopeOptions =
        optionsParser.getOptions(UniverseScopeOptions.class);
    if (cqueryOptions == null
        || universeScopeOptions == null
        || !isCqueryMode(cqueryOptions, universeScopeOptions)) {
      return;
    }
    // The post-analysis cquery filter requires a completed analysis phase before execution, so
    // analysis and execution can't be merged (Skymeld).
    try {
      optionsParser.parse(
          PriorityCategory.SOFTWARE_REQUIREMENT,
          "build --cquery requires sequential analysis and execution phases",
          ImmutableList.of("--noexperimental_merged_skyframe_analysis_execution"));
    } catch (OptionsParsingException e) {
      throw new IllegalStateException("build --cquery option failed to parse", e);
    }
  }

  private static boolean isCqueryMode(
      BuildCqueryOptions cqueryOptions, UniverseScopeOptions universeScopeOptions) {
    return !cqueryOptions.getCquery().isEmpty()
        || !universeScopeOptions.getUniverseScope().isEmpty();
  }

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    BlazeRuntime runtime = env.getRuntime();
    BuildCqueryOptions cqueryOptions = options.getOptions(BuildCqueryOptions.class);
    UniverseScopeOptions universeScopeOptions = options.getOptions(UniverseScopeOptions.class);

    if (isCqueryMode(cqueryOptions, universeScopeOptions)) {
      return execCquery(env, options, cqueryOptions, universeScopeOptions);
    }

    List<String> targets;
    try {
      targets = TargetPatternsHelper.readFrom(env, options);
    } catch (TargetPatternsHelper.TargetPatternsHelperException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return BlazeCommandResult.failureDetail(e.getFailureDetail());
    }
    if (targets.isEmpty()) {
      env.getReporter()
          .handle(
              Event.warn(
                  "Usage: "
                      + runtime.getProductName()
                      + " build <options> <targets>."
                      + "\nInvoke `"
                      + runtime.getProductName()
                      + " help build` for full description of usage and options."
                      + "\nYour request is correct, but requested an empty set of targets."
                      + " Nothing will be built."));
    }

    BuildRequest request;
    try (SilentCloseable closeable = Profiler.instance().profile("BuildRequest.create")) {

      request =
          BuildRequest.builder()
              .setCommandName(getClass().getAnnotation(Command.class).name())
              .setId(env.getCommandId())
              .setOptions(options)
              .setStartupOptions(runtime.getStartupOptionsProvider())
              .setOutErr(env.getReporter().getOutErr())
              .setTargets(targets)
              .setStartTimeMillis(env.getCommandStartTime())
              .build();
    }
    DetailedExitCode detailedExitCode =
        new BuildTool(env).processRequest(request, null, options).getDetailedExitCode();
    return BlazeCommandResult.detailedExitCode(detailedExitCode);
  }

  /**
   * Handles {@code build --cquery} / {@code build --universe_scope}: analyzes a universe, evaluates
   * a cquery over the resulting configured-target graph, and builds only the matched configured
   * targets.
   */
  private BlazeCommandResult execCquery(
      CommandEnvironment env,
      OptionsParsingResult options,
      BuildCqueryOptions cqueryOptions,
      UniverseScopeOptions universeScopeOptions) {
    BlazeRuntime runtime = env.getRuntime();
    String cqueryExpression = cqueryOptions.getCquery();
    List<String> universeScope = universeScopeOptions.getUniverseScope();
    List<String> residue = options.getResidue();

    if (!options.getOptions(BuildRequestOptions.class).getTargetPatternFile().isEmpty()) {
      return failure(
          env, "--cquery/--universe_scope cannot be combined with --target_pattern_file");
    }

    TargetPattern.Parser mainRepoTargetParser;
    try {
      RepositoryMapping repoMapping =
          env.getSkyframeExecutor()
              .getMainRepoMapping(
                  options.getOptions(KeepGoingOption.class).getKeepGoing(),
                  options.getOptions(LoadingPhaseThreadsOption.class).getThreads(),
                  env.getReporter());
      mainRepoTargetParser =
          new Parser(env.getRelativeWorkingDirectory(), RepositoryName.MAIN, repoMapping);
    } catch (RepositoryMappingResolutionException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return BlazeCommandResult.detailedExitCode(e.getDetailedExitCode());
    } catch (InterruptedException e) {
      String errorMessage = "Interrupted while resolving repository mapping for --cquery";
      env.getReporter().handle(Event.error(errorMessage));
      return BlazeCommandResult.detailedExitCode(
          InterruptedFailureDetails.detailedExitCode(errorMessage));
    }

    // Build the effective expression to evaluate over the analyzed universe:
    //  - --cquery alone        -> the cquery expression
    //  - residue alone         -> union of the residue labels
    //  - both                  -> (cquery) intersect (union of residue labels)
    //  - neither (scope only)  -> union of the universe-scope labels (build the whole universe)
    boolean hasCquery = !cqueryExpression.isEmpty();
    boolean hasResidue = !residue.isEmpty();
    String effectiveExpression;
    if (hasCquery && hasResidue) {
      effectiveExpression = "(" + cqueryExpression + ") intersect (" + union(residue) + ")";
    } else if (hasCquery) {
      effectiveExpression = cqueryExpression;
    } else if (hasResidue) {
      effectiveExpression = union(residue);
    } else {
      effectiveExpression = union(universeScope);
    }

    QueryExpression expr;
    try {
      expr = QueryParser.parse(effectiveExpression, cqueryFunctions(env));
    } catch (QuerySyntaxException e) {
      return failure(
          env,
          String.format(
              "Error while parsing --cquery '%s': %s",
              QueryExpression.truncate(effectiveExpression), e.getMessage()));
    }

    List<String> universeTargets;
    if (!universeScope.isEmpty()) {
      universeTargets = ImmutableList.copyOf(universeScope);
    } else {
      LinkedHashSet<String> patterns = new LinkedHashSet<>();
      expr.collectTargetPatterns(patterns);
      universeTargets = ImmutableList.copyOf(patterns);
    }

    BuildRequest request;
    try (SilentCloseable closeable = Profiler.instance().profile("BuildRequest.create")) {
      request =
          BuildRequest.builder()
              .setCommandName(getClass().getAnnotation(Command.class).name())
              .setId(env.getCommandId())
              .setOptions(options)
              .setStartupOptions(runtime.getStartupOptionsProvider())
              .setOutErr(env.getReporter().getOutErr())
              .setTargets(universeTargets)
              .setStartTimeMillis(env.getCommandStartTime())
              .build();
    }
    QueryCommandUtils.resetDeserializedKeysFromRemoteAnalysisCache(env);
    DetailedExitCode detailedExitCode =
        new BuildTool(env, new BuildCqueryProcessor(expr, mainRepoTargetParser))
            .processRequest(request, /* validator= */ null, options)
            .getDetailedExitCode();
    return BlazeCommandResult.detailedExitCode(detailedExitCode);
  }

  /** Joins target patterns into an additive query union expression. */
  private static String union(List<String> patterns) {
    return String.join(" + ", patterns);
  }

  private static HashMap<String, QueryFunction> cqueryFunctions(CommandEnvironment env) {
    HashMap<String, QueryFunction> functions = new HashMap<>();
    for (QueryFunction queryFunction : ConfiguredTargetQueryEnvironment.FUNCTIONS) {
      functions.put(queryFunction.getName(), queryFunction);
    }
    for (QueryFunction queryFunction : env.getRuntime().getQueryFunctions()) {
      functions.put(queryFunction.getName(), queryFunction);
    }
    return functions;
  }

  private static BlazeCommandResult failure(CommandEnvironment env, String message) {
    env.getReporter().handle(Event.error(message));
    return BlazeCommandResult.failureDetail(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setConfigurableQuery(
                ConfigurableQuery.newBuilder().setCode(Code.EXPRESSION_PARSE_FAILURE))
            .build());
  }
}
