// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BuildView.AnalysisResult;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.query2.ConfiguredTargetQueryEnvironment;
import com.google.devtools.build.lib.query2.CqueryThreadsafeCallback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.TargetLiteral;
import com.google.devtools.build.lib.query2.output.CqueryOptions;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.SkyframeExecutorWrappingWalkableGraph;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

/** A version of {@link BuildTool} that handles all cquery work. */
public class CqueryBuildTool extends BuildTool {

  private final QueryExpression queryExpression;

  public CqueryBuildTool(CommandEnvironment env, QueryExpression queryExpression) {
    super(env);
    this.queryExpression = queryExpression;
  }

  @Override
  protected void postProcessAnalysisResult(
      BuildRequest request,
      AnalysisResult analysisResult,
      BuildConfigurationCollection configurations)
      throws InterruptedException, ViewCreationFailedException,
          ConfiguredTargetQueryCommandLineException {
    // TODO: b/71905538 - this query will operate over the graph as constructed by analysis, but
    // will also pick up any nodes that are in the graph from prior builds. This makes the results
    // not reproducible at the level of a single command. Either tolerate, or wipe the analysis
    // graph beforehand if this option is specified, or add another option to wipe if desired
    // (SkyframeExecutor#handleConfiguredTargetChange should be sufficient).
    if (queryExpression != null) {
      if (!env.getSkyframeExecutor().tracksStateForIncrementality()) {
        throw new ConfiguredTargetQueryCommandLineException(
            "Configured query is not allowed if incrementality state is not being kept");
      }
      try {
        doConfiguredTargetQuery(
            request,
            configurations.getHostConfiguration(),
            analysisResult.getTopLevelTargetsWithConfigs(),
            queryExpression);
      } catch (QueryException | IOException e) {
        if (!request.getKeepGoing()) {
          throw new ViewCreationFailedException("Error doing configured target query", e);
        }
        env.getReporter().error(null, "Error doing configured target query", e);
      }
    }
  }

  private void doConfiguredTargetQuery(
      BuildRequest request,
      BuildConfiguration hostConfiguration,
      List<TargetAndConfiguration> topLevelTargetsWithConfigs,
      QueryExpression queryExpression)
      throws InterruptedException, QueryException, IOException {

    // Currently, CTQE assumes that all top level targets take on the same default config and we
    // don't have the ability to map multiple configs to multiple top level targets.
    // So for now, we only allow multiple targets when they all carry the same config.
    // TODO: b/71508373 - fully support multiple top level targets
    List<TargetAndConfiguration> nonNullTargets =
        topLevelTargetsWithConfigs
            .stream()
            .filter(targetAndConfig -> targetAndConfig.getConfiguration() != null)
            .collect(Collectors.toList());
    BuildConfiguration targetConfig = null;
    if (!nonNullTargets.isEmpty()) {
      targetConfig = nonNullTargets.get(0).getConfiguration();
      for (TargetAndConfiguration targAndConfig : topLevelTargetsWithConfigs) {
        if (targAndConfig.getConfiguration() != null
            && !targAndConfig.getConfiguration().equals(targetConfig)) {
          throw new QueryException(
              new TargetLiteral(queryExpression.toString()),
              String.format(
                  "Top-level targets %s and %s have different configurations (top-level "
                      + "targets with different configurations is not supported)",
                  nonNullTargets.get(0).getLabel(), targAndConfig.getLabel()));
        }
      }
    }
    WalkableGraph walkableGraph =
        SkyframeExecutorWrappingWalkableGraph.of(env.getSkyframeExecutor());
    ImmutableList<QueryFunction> extraFunctions =
        new ImmutableList.Builder<QueryFunction>()
            .addAll(ConfiguredTargetQueryEnvironment.CQUERY_FUNCTIONS)
            .addAll(env.getRuntime().getQueryFunctions())
            .build();
    CqueryOptions cqueryOptions = request.getOptions(CqueryOptions.class);
    ConfiguredTargetQueryEnvironment configuredTargetQueryEnvironment =
        new ConfiguredTargetQueryEnvironment(
            request.getKeepGoing(),
            env.getReporter(),
            extraFunctions,
            targetConfig,
            hostConfiguration,
            env.newTargetPatternEvaluator().getOffset(),
            env.getPackageManager().getPackagePath(),
            () -> walkableGraph,
            cqueryOptions.toSettings());
    Iterable<CqueryThreadsafeCallback> callbacks =
        configuredTargetQueryEnvironment.getDefaultOutputFormatters(
            configuredTargetQueryEnvironment.getAccessor(),
            cqueryOptions,
            env.getReporter(),
            env.getSkyframeExecutor(),
            hostConfiguration,
            runtime.getRuleClassProvider().getTrimmingTransitionFactory(),
            cqueryOptions.aspectDeps.createResolver(env.getPackageManager(), env.getReporter()));
    CqueryThreadsafeCallback callback =
        CqueryThreadsafeCallback.getCallback(cqueryOptions.outputFormat, callbacks);
    if (callback == null) {
      env.getReporter()
          .handle(
              Event.error(
                  String.format(
                      "Invalid output format '%s'. Valid values are: %s",
                      cqueryOptions.outputFormat,
                      CqueryThreadsafeCallback.callbackNames(callbacks))));
      return;
    }
    QueryEvalResult result =
        configuredTargetQueryEnvironment.evaluateQuery(queryExpression, callback);
    if (result.isEmpty()) {
      env.getReporter().handle(Event.info("Empty query results"));
    }
  }

  /** Cquery specific command line exception. */
  protected static class ConfiguredTargetQueryCommandLineException extends Exception {
    ConfiguredTargetQueryCommandLineException(String message) {
      super(message);
    }
  }
}
