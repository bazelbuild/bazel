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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment.TopLevelConfigurations;
import com.google.devtools.build.lib.query2.cquery.ConfiguredTargetQueryEnvironment;
import com.google.devtools.build.lib.query2.cquery.CqueryOptions;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.skyframe.WalkableGraph;

/** A version of {@link BuildTool} that handles all cquery work. */
public final class CqueryBuildTool extends PostAnalysisQueryBuildTool<ConfiguredTarget> {

  public CqueryBuildTool(CommandEnvironment env, QueryExpression queryExpression) {
    super(env, queryExpression);
  }

  @Override
  protected ConfiguredTargetQueryEnvironment getQueryEnvironment(
      BuildRequest request,
      BuildConfiguration hostConfiguration,
      TopLevelConfigurations configurations,
      WalkableGraph walkableGraph) {
    ImmutableList<QueryFunction> extraFunctions =
        new ImmutableList.Builder<QueryFunction>()
            .addAll(ConfiguredTargetQueryEnvironment.CQUERY_FUNCTIONS)
            .addAll(env.getRuntime().getQueryFunctions())
            .build();
    CqueryOptions cqueryOptions = request.getOptions(CqueryOptions.class);
    return new ConfiguredTargetQueryEnvironment(
        request.getKeepGoing(),
        env.getReporter(),
        extraFunctions,
        configurations,
        hostConfiguration,
        env.getRelativeWorkingDirectory().getPathString(),
        env.getPackageManager().getPackagePath(),
        () -> walkableGraph,
        cqueryOptions);
  }
}
