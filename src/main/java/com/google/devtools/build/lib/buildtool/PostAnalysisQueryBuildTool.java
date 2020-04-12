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
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.query2.NamedThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment.TopLevelConfigurations;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.QueryRuntimeHelper;
import com.google.devtools.build.lib.skyframe.SkyframeExecutorWrappingWalkableGraph;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.IOException;
import java.util.Collection;

/**
 * Version of {@link BuildTool} that handles all work for queries based on results from the analysis
 * phase.
 */
public abstract class PostAnalysisQueryBuildTool<T> extends BuildTool {

  private final QueryExpression queryExpression;

  public PostAnalysisQueryBuildTool(CommandEnvironment env, QueryExpression queryExpression) {
    super(env);
    this.queryExpression = queryExpression;
  }

  @Override
  protected void postProcessAnalysisResult(BuildRequest request, AnalysisResult analysisResult)
      throws InterruptedException, ViewCreationFailedException, QueryCommandLineException {
    // TODO: b/71905538 - this query will operate over the graph as constructed by analysis, but
    // will also pick up any nodes that are in the graph from prior builds. This makes the results
    // not reproducible at the level of a single command. Either tolerate, or wipe the analysis
    // graph beforehand if this option is specified, or add another option to wipe if desired
    // (SkyframeExecutor#handleAnalysisInvalidatingChange should be sufficient).
    if (queryExpression != null) {
      if (!env.getSkyframeExecutor().tracksStateForIncrementality()) {
        throw new QueryCommandLineException(
            "Queries based on analysis results are not allowed "
                + "if incrementality state is not being kept");
      }
      try (QueryRuntimeHelper queryRuntimeHelper =
          env.getRuntime().getQueryRuntimeHelperFactory().create(env)) {
        doPostAnalysisQuery(
            request,
            analysisResult.getConfigurationCollection().getHostConfiguration(),
            new TopLevelConfigurations(analysisResult.getTopLevelTargetsWithConfigs()),
            env.getSkyframeExecutor().getTransitiveConfigurationKeys(),
            queryRuntimeHelper,
            queryExpression);
      } catch (QueryException | IOException e) {
        if (!request.getKeepGoing()) {
          throw new ViewCreationFailedException("Error doing post analysis query", e);
        }
        env.getReporter().error(null, "Error doing post analysis query", e);
      } catch (QueryRuntimeHelper.Factory.CommandLineException e) {
        throw new QueryCommandLineException(e.getMessage());
      }
    }
  }

  protected abstract PostAnalysisQueryEnvironment<T> getQueryEnvironment(
      BuildRequest request,
      BuildConfiguration hostConfiguration,
      TopLevelConfigurations topLevelConfigurations,
      Collection<SkyKey> transitiveConfigurationKeys,
      WalkableGraph walkableGraph)
      throws InterruptedException;

  private void doPostAnalysisQuery(
      BuildRequest request,
      BuildConfiguration hostConfiguration,
      TopLevelConfigurations topLevelConfigurations,
      Collection<SkyKey> transitiveConfigurationKeys,
      QueryRuntimeHelper queryRuntimeHelper,
      QueryExpression queryExpression)
      throws InterruptedException, QueryException, IOException {
    WalkableGraph walkableGraph =
        SkyframeExecutorWrappingWalkableGraph.of(env.getSkyframeExecutor());

    PostAnalysisQueryEnvironment<T> postAnalysisQueryEnvironment =
        getQueryEnvironment(
            request,
            hostConfiguration,
            topLevelConfigurations,
            transitiveConfigurationKeys,
            walkableGraph);

    Iterable<NamedThreadSafeOutputFormatterCallback<T>> callbacks =
        postAnalysisQueryEnvironment.getDefaultOutputFormatters(
            postAnalysisQueryEnvironment.getAccessor(),
            env.getReporter(),
            queryRuntimeHelper.getOutputStreamForQueryOutput(),
            env.getSkyframeExecutor(),
            hostConfiguration,
            runtime.getRuleClassProvider().getTrimmingTransitionFactory(),
            env.getPackageManager());
    String outputFormat = postAnalysisQueryEnvironment.getOutputFormat();
    NamedThreadSafeOutputFormatterCallback<T> callback =
        NamedThreadSafeOutputFormatterCallback.selectCallback(outputFormat, callbacks);
    if (callback == null) {
      env.getReporter()
          .handle(
              Event.error(
                  String.format(
                      "Invalid output format '%s'. Valid values are: %s",
                      outputFormat,
                      NamedThreadSafeOutputFormatterCallback.callbackNames(callbacks))));
      return;
    }
    QueryEvalResult result =
        postAnalysisQueryEnvironment.evaluateQuery(queryExpression, callback);
    if (result.isEmpty()) {
      env.getReporter().handle(Event.info("Empty query results"));
    }
    queryRuntimeHelper.afterQueryOutputIsWritten();
  }
}
