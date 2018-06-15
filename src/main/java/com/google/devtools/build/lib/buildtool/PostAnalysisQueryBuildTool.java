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

import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.query2.ConfiguredTargetQueryEnvironment;
import com.google.devtools.build.lib.query2.CqueryThreadsafeCallback;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.TargetLiteral;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.SkyframeExecutorWrappingWalkableGraph;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Version of {@link BuildTool} that handles all work for queries based on results from the analysis
 * phase.
 */
public abstract class PostAnalysisQueryBuildTool extends BuildTool {

  private final QueryExpression queryExpression;

  public PostAnalysisQueryBuildTool(CommandEnvironment env, QueryExpression queryExpression) {
    super(env);
    this.queryExpression = queryExpression;
  }

  @Override
  protected void postProcessAnalysisResult(
      BuildRequest request,
      AnalysisResult analysisResult)
      throws InterruptedException, ViewCreationFailedException,
          PostAnalysisQueryCommandLineException {
    // TODO: b/71905538 - this query will operate over the graph as constructed by analysis, but
    // will also pick up any nodes that are in the graph from prior builds. This makes the results
    // not reproducible at the level of a single command. Either tolerate, or wipe the analysis
    // graph beforehand if this option is specified, or add another option to wipe if desired
    // (SkyframeExecutor#handleConfiguredTargetChange should be sufficient).
    if (queryExpression != null) {
      if (!env.getSkyframeExecutor().tracksStateForIncrementality()) {
        throw new PostAnalysisQueryCommandLineException(
            "Queries based on analysis results are not allowed "
                + "if incrementality state is not being kept");
      }
      try {
        doPostAnalysisQuery(
            request,
            analysisResult.getConfigurationCollection().getHostConfiguration(),
            analysisResult.getTopLevelTargetsWithConfigs(),
            queryExpression);
      } catch (QueryException | IOException e) {
        if (!request.getKeepGoing()) {
          throw new ViewCreationFailedException("Error doing post analysis query", e);
        }
        env.getReporter().error(null, "Error doing post analysis query", e);
      }
    }
  }

  // TODO(twerth): Make this more generic when introducting a PostAnalysisQueryEnvironment.
  protected abstract ConfiguredTargetQueryEnvironment getQueryEnvironment(
      BuildRequest request,
      BuildConfiguration hostConfiguration,
      BuildConfiguration targetConfig,
      WalkableGraph walkableGraph);

  private BuildConfiguration getBuildConfiguration(
      List<TargetAndConfiguration> topLevelTargetsWithConfigs, QueryExpression queryExpression)
      throws QueryException {
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
    return targetConfig;
  }

  private void doPostAnalysisQuery(
      BuildRequest request,
      BuildConfiguration hostConfiguration,
      List<TargetAndConfiguration> topLevelTargetsWithConfigs,
      QueryExpression queryExpression)
      throws InterruptedException, QueryException, IOException {
    BuildConfiguration targetConfig =
        getBuildConfiguration(topLevelTargetsWithConfigs, queryExpression);

    WalkableGraph walkableGraph =
        SkyframeExecutorWrappingWalkableGraph.of(env.getSkyframeExecutor());

    ConfiguredTargetQueryEnvironment configuredTargetQueryEnvironment =
        getQueryEnvironment(request, hostConfiguration, targetConfig, walkableGraph);
    Iterable<CqueryThreadsafeCallback> callbacks =
        configuredTargetQueryEnvironment.getDefaultOutputFormatters(
            configuredTargetQueryEnvironment.getAccessor(),
            env.getReporter(),
            env.getSkyframeExecutor(),
            hostConfiguration,
            runtime.getRuleClassProvider().getTrimmingTransitionFactory(),
            env.getPackageManager());
    String outputFormat = configuredTargetQueryEnvironment.getOutputFormat();
    CqueryThreadsafeCallback callback =
        CqueryThreadsafeCallback.getCallback(outputFormat, callbacks);
    if (callback == null) {
      env.getReporter()
          .handle(
              Event.error(
                  String.format(
                      "Invalid output format '%s'. Valid values are: %s",
                      outputFormat, CqueryThreadsafeCallback.callbackNames(callbacks))));
      return;
    }
    QueryEvalResult result =
        configuredTargetQueryEnvironment.evaluateQuery(queryExpression, callback);
    if (result.isEmpty()) {
      env.getReporter().handle(Event.info("Empty query results"));
    }
  }

  /** Post analysis query specific command line exception. */
  protected static class PostAnalysisQueryCommandLineException extends Exception {
    PostAnalysisQueryCommandLineException(String message) {
      super(message);
    }
  }
}
