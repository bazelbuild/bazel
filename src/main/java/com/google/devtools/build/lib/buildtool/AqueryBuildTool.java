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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.query2.ActionGraphQueryEnvironment;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment.TopLevelConfigurations;
import com.google.devtools.build.lib.query2.engine.ActionFilterFunction;
import com.google.devtools.build.lib.query2.engine.FunctionExpression;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.output.AqueryOptions;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.Optional;
import java.util.regex.Pattern;

/** A version of {@link BuildTool} that handles all aquery work. */
public class AqueryBuildTool extends PostAnalysisQueryBuildTool<ConfiguredTargetValue> {
  private final ImmutableMap<String, Pattern> actionFilters;

  public AqueryBuildTool(CommandEnvironment env, QueryExpression queryExpression)
      throws AqueryActionFilterException {
    super(env, queryExpression);
    actionFilters = getActionFilters(queryExpression);
  }

  @Override
  protected PostAnalysisQueryEnvironment<ConfiguredTargetValue> getQueryEnvironment(
      BuildRequest request,
      BuildConfiguration hostConfiguration,
      TopLevelConfigurations topLevelConfigurations,
      WalkableGraph walkableGraph) {
    ImmutableList<QueryFunction> extraFunctions =
        new ImmutableList.Builder<QueryFunction>()
            .addAll(ActionGraphQueryEnvironment.AQUERY_FUNCTIONS)
            .addAll(env.getRuntime().getQueryFunctions())
            .build();
    AqueryOptions aqueryOptions = request.getOptions(AqueryOptions.class);

    ActionGraphQueryEnvironment queryEnvironment =
        new ActionGraphQueryEnvironment(
            request.getKeepGoing(),
            env.getReporter(),
            extraFunctions,
            topLevelConfigurations,
            hostConfiguration,
            env.getRelativeWorkingDirectory().getPathString(),
            env.getPackageManager().getPackagePath(),
            () -> walkableGraph,
            aqueryOptions);
    queryEnvironment.setActionFilters(actionFilters);

    return queryEnvironment;
  }

  /**
   * Return the action filters in the form { inputs: <pattern>, outputs: <pattern>, ... }
   *
   * @param queryExpression The query expression from aquery command
   * @return the action filters
   * @throws AqueryActionFilterException if an aquery filter function is preceded by any other
   *     function types
   */
  private ImmutableMap<String, Pattern> getActionFilters(QueryExpression queryExpression)
      throws AqueryActionFilterException {
    ImmutableMap.Builder<String, Pattern> actionFiltersBuilder = ImmutableMap.builder();

    if (!(queryExpression instanceof FunctionExpression)) {
      return actionFiltersBuilder.build();
    }

    Optional<FunctionExpression> functionExpressionOptional =
        Optional.of((FunctionExpression) queryExpression);

    FunctionExpression nonAqueryFilterFunctionExpression = null;

    // Unwrap the function layers
    // Validate that aquery filter functions (inputs, outputs, mnemonics) are not preceded
    // by any other function types
    while (functionExpressionOptional.isPresent()) {
      FunctionExpression functionExpression = functionExpressionOptional.get();

      if (functionExpression.getFunction() instanceof ActionFilterFunction) {
        if (nonAqueryFilterFunctionExpression != null) {
          throw new AqueryActionFilterException(
              "aquery filter functions (inputs, outputs, mnemonic) produce actions, and therefore "
                  + "can't be the input of other function types: "
                  + nonAqueryFilterFunctionExpression.getFunction().getName());
        }
        ActionFilterFunction actionFilterFunction =
            (ActionFilterFunction) functionExpression.getFunction();

        // TODO(leba) support multiple patterns for 1 function type
        String patternString = functionExpression.getArgs().get(0).getWord();
        actionFiltersBuilder.put(actionFilterFunction.getName(), Pattern.compile(patternString));
      } else {
        nonAqueryFilterFunctionExpression = functionExpression;
      }

      functionExpressionOptional = getNextFunctionExpression(functionExpression);
    }

    return actionFiltersBuilder.build();
  }

  /**
   * Unwrap input {@code functionExpression} to get the next FunctionExpression in the query
   *
   * @param functionExpression the current function expression
   * @return the Optional of the next FunctionExpression in the query
   */
  private Optional<FunctionExpression> getNextFunctionExpression(
      FunctionExpression functionExpression) {
    for (Argument arg : functionExpression.getArgs()) {
      if (arg.getType() == ArgumentType.EXPRESSION
          && arg.getExpression() instanceof FunctionExpression) {
        return Optional.of((FunctionExpression) arg.getExpression());
      }
    }
    return Optional.empty();
  }

  /** Custom exception class for aquery filtering */
  public static class AqueryActionFilterException extends Exception {
    AqueryActionFilterException(String message) {
      super(message);
    }
  }
}
