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
package com.google.devtools.build.lib.runtime.commands;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.CoreOptions.IncludeConfigFragmentsEnum;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.CqueryBuildTool;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.query2.cquery.ConfiguredTargetQueryEnvironment;
import com.google.devtools.build.lib.query2.cquery.CqueryOptions;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.query2.engine.QuerySyntaxException;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery;
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/** Handles the 'cquery' command on the Blaze command line. */
@Command(
  name = "cquery",
  builds = true,
  inherits = {BuildCommand.class},
  options = {CqueryOptions.class},
  usesConfigurationOptions = true,
  shortDescription = "Loads, analyzes, and queries the specified targets w/ configurations.",
  allowResidue = true,
  completion = "label",
  help = "resource:cquery.txt"
)
public final class CqueryCommand implements BlazeCommand {

  @Override
  public void editOptions(OptionsParser optionsParser) {
    CqueryOptions cqueryOptions = optionsParser.getOptions(CqueryOptions.class);
    try {
      if (!cqueryOptions.transitions.equals(CqueryOptions.Transitions.NONE)) {
        optionsParser.parse(
            PriorityCategory.COMPUTED_DEFAULT,
            "Option required by setting the --transitions flag",
            ImmutableList.of("--output=transitions"));
      }
      optionsParser.parse(
          PriorityCategory.COMPUTED_DEFAULT,
          "Options required by cquery",
          ImmutableList.of("--nobuild"));
      optionsParser.parse(
          PriorityCategory.COMPUTED_DEFAULT,
          "cquery should include 'tags = [\"manual\"]' targets by default",
          ImmutableList.of("--build_manual_tests"));
      optionsParser.parse(
          PriorityCategory.COMPUTED_DEFAULT,
          // https://github.com/bazelbuild/bazel/issues/11078
          "cquery should not exclude test_suite rules",
          ImmutableList.of("--noexpand_test_suites"));
      if (cqueryOptions.showRequiredConfigFragments != IncludeConfigFragmentsEnum.OFF) {
        optionsParser.parse(
            PriorityCategory.COMPUTED_DEFAULT,
            "Options required by cquery's --show_config_fragments flag",
            ImmutableList.of(
                "--include_config_fragments_provider="
                    + cqueryOptions.showRequiredConfigFragments));
      }
    } catch (OptionsParsingException e) {
      throw new IllegalStateException("Cquery's known options failed to parse", e);
    }
  }

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    if (options.getResidue().isEmpty()) {
      String message =
          "Missing query expression. Use the 'help cquery' command for syntax and help.";
      env.getReporter().handle(Event.error(message));
      return createFailureResult(message, Code.COMMAND_LINE_EXPRESSION_MISSING);
    }
    String query = Joiner.on(' ').join(options.getResidue());
    HashMap<String, QueryFunction> functions = new HashMap<>();
    for (QueryFunction queryFunction : ConfiguredTargetQueryEnvironment.FUNCTIONS) {
      functions.put(queryFunction.getName(), queryFunction);
    }
    for (QueryFunction queryFunction : env.getRuntime().getQueryFunctions()) {
      functions.put(queryFunction.getName(), queryFunction);
    }
    QueryExpression expr;
    try {
      expr = QueryParser.parse(query, functions);
    } catch (QuerySyntaxException e) {
      String message =
          String.format(
              "Error while parsing '%s': %s", QueryExpression.truncate(query), e.getMessage());
      env.getReporter().handle(Event.error(message));
      return createFailureResult(message, Code.EXPRESSION_PARSE_FAILURE);
    }

    List<String> topLevelTargets = options.getOptions(CqueryOptions.class).universeScope;
    Set<String> targetPatternSet = new LinkedHashSet<>();
    if (topLevelTargets.isEmpty()) {
      expr.collectTargetPatterns(targetPatternSet);
      topLevelTargets = new ArrayList<>(targetPatternSet);
    }
    BlazeRuntime runtime = env.getRuntime();

    BuildRequest request =
        BuildRequest.builder()
            .setCommandName(getClass().getAnnotation(Command.class).name())
            .setId(env.getCommandId())
            .setOptions(options)
            .setStartupOptions(runtime.getStartupOptionsProvider())
            .setOutErr(env.getReporter().getOutErr())
            .setTargets(topLevelTargets)
            .setStartTimeMillis(env.getCommandStartTime())
            .build();
    DetailedExitCode detailedExitCode =
        new CqueryBuildTool(env, expr).processRequest(request, null).getDetailedExitCode();
    return BlazeCommandResult.detailedExitCode(detailedExitCode);
  }

  private static BlazeCommandResult createFailureResult(String message, Code detailedCode) {
    return BlazeCommandResult.failureDetail(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setConfigurableQuery(ConfigurableQuery.newBuilder().setCode(detailedCode))
            .build());
  }
}
