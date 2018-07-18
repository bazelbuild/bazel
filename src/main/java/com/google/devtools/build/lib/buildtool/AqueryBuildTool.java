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
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.query2.ActionGraphQueryEnvironment;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.output.AqueryOptions;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.skyframe.WalkableGraph;

/** A version of {@link BuildTool} that handles all aquery work. */
public class AqueryBuildTool extends PostAnalysisQueryBuildTool<ConfiguredTargetValue> {

  public AqueryBuildTool(CommandEnvironment env, QueryExpression queryExpression) {
    super(env, queryExpression);
  }

  @Override
  protected PostAnalysisQueryEnvironment<ConfiguredTargetValue> getQueryEnvironment(
      BuildRequest request,
      BuildConfiguration hostConfiguration,
      BuildConfiguration targetConfig,
      WalkableGraph walkableGraph) {
    ImmutableList<QueryFunction> extraFunctions =
        new ImmutableList.Builder<QueryFunction>()
            .addAll(ActionGraphQueryEnvironment.AQUERY_FUNCTIONS)
            .addAll(env.getRuntime().getQueryFunctions())
            .build();
    AqueryOptions aqueryOptions = request.getOptions(AqueryOptions.class);
    return new ActionGraphQueryEnvironment(
        request.getKeepGoing(),
        env.getReporter(),
        extraFunctions,
        targetConfig,
        hostConfiguration,
        env.getRelativeWorkingDirectory().getPathString(),
        env.getPackageManager().getPackagePath(),
        () -> walkableGraph,
        aqueryOptions);
  }
}
