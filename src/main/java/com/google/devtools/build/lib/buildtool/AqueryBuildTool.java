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
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.analysis.AnalysisProtos.ActionGraphContainer;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment.TopLevelConfigurations;
import com.google.devtools.build.lib.query2.aquery.ActionGraphProtoV2OutputFormatterCallback;
import com.google.devtools.build.lib.query2.aquery.ActionGraphQueryEnvironment;
import com.google.devtools.build.lib.query2.aquery.AqueryActionFilter;
import com.google.devtools.build.lib.query2.aquery.AqueryOptions;
import com.google.devtools.build.lib.query2.engine.ActionFilterFunction;
import com.google.devtools.build.lib.query2.engine.FunctionExpression;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.QueryRuntimeHelper;
import com.google.devtools.build.lib.runtime.QueryRuntimeHelper.Factory.CommandLineException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.ActionGraphDump;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryOutputHandler;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryOutputHandler.OutputType;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Collection;
import java.util.Optional;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import javax.annotation.Nullable;

/** A version of {@link BuildTool} that handles all aquery work. */
public final class AqueryBuildTool extends PostAnalysisQueryBuildTool<ConfiguredTargetValue> {
  private final AqueryActionFilter actionFilters;

  public AqueryBuildTool(CommandEnvironment env, @Nullable QueryExpression queryExpression)
      throws AqueryActionFilterException {
    super(env, queryExpression);
    actionFilters = buildActionFilters(queryExpression);
  }

  /** Outputs the current {@link ActionGraphContainer} of Skyframe. */
  public BlazeCommandResult dumpActionGraphFromSkyframe(BuildRequest request) {
    try (QueryRuntimeHelper queryRuntimeHelper =
        env.getRuntime().getQueryRuntimeHelperFactory().create(env)) {
      AqueryOptions aqueryOptions = request.getOptions(AqueryOptions.class);

      PrintStream printStream =
          queryRuntimeHelper.getOutputStreamForQueryOutput() == null
              ? null
              : new PrintStream(queryRuntimeHelper.getOutputStreamForQueryOutput());

      if (aqueryOptions.protoV2) {
        AqueryOutputHandler aqueryOutputHandler =
            ActionGraphProtoV2OutputFormatterCallback.constructAqueryOutputHandler(
                OutputType.fromString(aqueryOptions.outputFormat),
                queryRuntimeHelper.getOutputStreamForQueryOutput(),
                printStream);
        try {
          ActionGraphDump actionGraphDump =
              new ActionGraphDump(
                  aqueryOptions.includeCommandline,
                  aqueryOptions.includeArtifacts,
                  actionFilters,
                  aqueryOptions.includeParamFiles,
                  aqueryOutputHandler);
          ((SequencedSkyframeExecutor) env.getSkyframeExecutor())
              .dumpSkyframeState(actionGraphDump);
        } finally {
          aqueryOutputHandler.close();
        }
      } else {
        ActionGraphContainer actionGraphContainer =
            ((SequencedSkyframeExecutor) env.getSkyframeExecutor())
                .getActionGraphContainer(
                    aqueryOptions.includeCommandline,
                    actionFilters,
                    aqueryOptions.includeParamFiles,
                    aqueryOptions.includeArtifacts);

        // Write the data.
        if (OutputType.BINARY.formatName().equals(aqueryOptions.outputFormat)) {
          actionGraphContainer.writeTo(printStream);
        } else if (OutputType.TEXT.formatName().equals(aqueryOptions.outputFormat)) {
          TextFormat.printer().print(actionGraphContainer, printStream);
        } else {
          throw new IllegalStateException(
              "Unsupported output format "
                  + aqueryOptions.outputFormat
                  + ": --skyframe_state must be used with --output=proto or --output=textproto.");
        }
      }
      return BlazeCommandResult.exitCode(ExitCode.SUCCESS);
    } catch (CommandLineExpansionException | CommandLineException e) {
      env.getReporter().handle(Event.error("Error while parsing command: " + e.getMessage()));
      return BlazeCommandResult.exitCode(ExitCode.COMMAND_LINE_ERROR);
    } catch (IOException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return BlazeCommandResult.exitCode(ExitCode.RUN_FAILURE);
    }
  }

  @Override
  protected PostAnalysisQueryEnvironment<ConfiguredTargetValue> getQueryEnvironment(
      BuildRequest request,
      BuildConfiguration hostConfiguration,
      TopLevelConfigurations topLevelConfigurations,
      Collection<SkyKey> transitiveConfigurationKeys,
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
  private AqueryActionFilter buildActionFilters(@Nullable QueryExpression queryExpression)
      throws AqueryActionFilterException {
    AqueryActionFilter.Builder actionFiltersBuilder = AqueryActionFilter.builder();

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

        String patternString = functionExpression.getArgs().get(0).getWord();
        try {
          actionFiltersBuilder.put(actionFilterFunction.getName(), Pattern.compile(patternString));
        } catch (PatternSyntaxException e) {
          throw new AqueryActionFilterException("Wrong query syntax: " + e.getMessage());
        }
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
