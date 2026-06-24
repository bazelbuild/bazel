// Copyright 2026 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.buildtool.BuildTool.ExitException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment.TopLevelConfigurations;
import com.google.devtools.build.lib.query2.common.CommonQueryOptions;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.cquery.ConfiguredTargetQueryEnvironment;
import com.google.devtools.build.lib.query2.cquery.CqueryOptions;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.devtools.common.options.Options;
import java.io.IOException;
import java.util.Set;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * {@link BuildTool.AnalysisPostProcessor} for {@code build --cquery}: evaluates a cquery expression
 * over the post-analysis configured-target graph and returns a filtered {@link AnalysisResult}
 * containing only the matched configured targets, so the execution phase builds only those.
 *
 * <p>Unlike {@link CqueryProcessor} (used by the standalone {@code cquery} command), this processor
 * writes no query output to stdout; it uses the query result purely to restrict what gets built.
 *
 * <p>Code sharing with {@link CqueryProcessor}: both extend {@link PostAnalysisQueryProcessor} and
 * build the same {@link ConfiguredTargetQueryEnvironment}. Because the {@code build} command does
 * not register {@link CqueryOptions}, this processor falls back to default cquery options.
 */
public final class BuildCqueryProcessor extends PostAnalysisQueryProcessor<CqueryNode> {

  public BuildCqueryProcessor(
      QueryExpression queryExpression, TargetPattern.Parser mainRepoTargetParser) {
    super(queryExpression, mainRepoTargetParser);
  }

  private static CqueryOptions defaultCqueryOptions() {
    return Options.getDefaults(CqueryOptions.class);
  }

  @Override
  protected CommonQueryOptions getQueryOptions(CommandEnvironment env) {
    CqueryOptions options = env.getOptions().getOptions(CqueryOptions.class);
    return options != null ? options : defaultCqueryOptions();
  }

  @Override
  protected ConfiguredTargetQueryEnvironment getQueryEnvironment(
      BuildRequest request,
      CommandEnvironment env,
      TopLevelConfigurations configurations,
      ImmutableMap<String, BuildConfigurationValue> transitiveConfigurations,
      ImmutableMap<AspectKeyCreator.AspectKey, ConfiguredAspect> topLevelAspects,
      WalkableGraph walkableGraph) {
    ImmutableList<QueryFunction> extraFunctions =
        ImmutableList.<QueryFunction>builder()
            .addAll(ConfiguredTargetQueryEnvironment.CQUERY_FUNCTIONS)
            .addAll(env.getRuntime().getQueryFunctions())
            .build();
    CqueryOptions cqueryOptions = request.getOptions(CqueryOptions.class);
    if (cqueryOptions == null) {
      cqueryOptions = defaultCqueryOptions();
    }
    StarlarkSemantics starlarkSemantics =
        env.getSkyframeExecutor()
            .getEffectiveStarlarkSemantics(env.getOptions().getOptions(BuildLanguageOptions.class));
    return new ConfiguredTargetQueryEnvironment(
        request.getKeepGoing(),
        env.getReporter(),
        extraFunctions,
        configurations,
        transitiveConfigurations,
        topLevelAspects,
        mainRepoTargetParser,
        env.getPackageManager().getPackagePath(),
        () -> walkableGraph,
        cqueryOptions,
        request.getTopLevelArtifactContext(),
        cqueryOptions.getLabelPrinter(starlarkSemantics, mainRepoTargetParser.getRepoMapping()));
  }

  /**
   * Evaluates the cquery expression against the post-analysis graph and returns a copy of {@code
   * analysisResult} restricted to only the configured targets matched by the query.
   */
  @Override
  public AnalysisResult process(
      BuildRequest request,
      CommandEnvironment env,
      BlazeRuntime runtime,
      AnalysisResult analysisResult)
      throws InterruptedException, ViewCreationFailedException, ExitException {
    env.getSkyframeExecutor().deleteOldNodes(/* versionWindowForDirtyGc= */ 0);
    env.getSkyframeExecutor().applyInvalidation(env.getReporter());

    Set<CqueryNode> matchedNodes;
    try {
      matchedNodes =
          evaluateQueryNodes(
              request,
              env,
              new TopLevelConfigurations(analysisResult.getTopLevelTargetsWithConfigs()),
              analysisResult.getAspectsMap(),
              env.getSkyframeExecutor().getTransitiveConfigurationKeys(),
              queryExpression);
    } catch (QueryException e) {
      String errorMessage = "Error evaluating --cquery expression";
      if (!request.getKeepGoing()) {
        throw new ViewCreationFailedException(errorMessage, e.getFailureDetail(), e);
      }
      env.getReporter().error(null, errorMessage + ": " + e.getFailureDetail().getMessage());
      return analysisResult;
    } catch (IOException e) {
      FailureDetail failureDetail =
          FailureDetail.newBuilder()
              .setMessage("I/O error evaluating --cquery expression: " + e.getMessage())
              .setConfigurableQuery(
                  ConfigurableQuery.newBuilder()
                      .setCode(ConfigurableQuery.Code.CONFIGURABLE_QUERY_UNKNOWN))
              .build();
      throw new ExitException(DetailedExitCode.of(failureDetail));
    }

    ImmutableSet<ConfiguredTarget> filteredTargets =
        matchedNodes.stream()
            .filter(ConfiguredTarget.class::isInstance)
            .map(ConfiguredTarget.class::cast)
            .collect(ImmutableSet.toImmutableSet());

    return analysisResult.withFilteredTargets(filteredTargets);
  }
}
