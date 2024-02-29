// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.commands;

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.BuildTool;
import com.google.devtools.build.lib.buildtool.CqueryProcessor;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern.Parser;
import com.google.devtools.build.lib.query2.NamedThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.cquery.ConfiguredTargetQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.query2.engine.QuerySyntaxException;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue.RepositoryMappingResolutionException;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Stream;

/** Fetches all repos needed for building a given set of targets. */
public class TargetFetcher {
  private final CommandEnvironment env;

  private TargetFetcher(CommandEnvironment env) {
    this.env = env;
  }

  /** Uses cquery to find and fetch all repos needed to build these targets */
  public static void fetchTargets(
      CommandEnvironment env, OptionsParsingResult options, List<String> targets)
      throws RepositoryMappingResolutionException, InterruptedException, TargetFetcherException {
    new TargetFetcher(env).fetchTargets(options, targets);
  }

  private void fetchTargets(OptionsParsingResult options, List<String> targets)
      throws InterruptedException, TargetFetcherException, RepositoryMappingResolutionException {
    QueryExpression expr = createQueryExpression(targets);
    BuildRequest request = createBuildRequest(env, options, targets);
    TargetPattern.Parser mainRepoTargetParser = getMainRepoMappingParser(env);

    BuildResult result =
        new BuildTool(
                env,
                new CqueryProcessor(
                    expr, mainRepoTargetParser, Optional.of(createNoOutputFormatter())))
            .processRequest(request, /* validator= */ null);
    if (!result.getSuccess()) {
      throw new TargetFetcherException(
          String.format(
              "Fetching some target dependencies for %s failed, but --keep_going specified. "
                  + " Ignoring errors",
              expr));
    }
  }

  /** Creates special output formatter for fetch that doesn't print anything */
  private NamedThreadSafeOutputFormatterCallback<CqueryNode> createNoOutputFormatter() {
    return new NamedThreadSafeOutputFormatterCallback<CqueryNode>() {
      @Override
      public String getName() {
        return "no_output";
      }

      @Override
      public void processOutput(Iterable<CqueryNode> partialResult) {
        // Just do nothing!
        // This will be later used to collect repos for vendoring
      }
    };
  }

  private BuildRequest createBuildRequest(
      CommandEnvironment env, OptionsParsingResult options, List<String> targets) {
    return BuildRequest.builder()
        .setCommandName(env.getCommandName())
        .setId(env.getCommandId())
        .setOptions(options)
        .setStartupOptions(env.getRuntime().getStartupOptionsProvider())
        .setOutErr(env.getReporter().getOutErr())
        .setTargets(targets)
        .setStartTimeMillis(env.getCommandStartTime())
        .setCheckforActionConflicts(false)
        .setReportIncompatibleTargets(false)
        .build();
  }

  private Parser getMainRepoMappingParser(CommandEnvironment env)
      throws RepositoryMappingResolutionException, InterruptedException {
    RepositoryMapping repoMapping =
        env.getSkyframeExecutor()
            .getMainRepoMapping(
                env.getOptions().getOptions(KeepGoingOption.class).keepGoing,
                env.getOptions().getOptions(LoadingPhaseThreadsOption.class).threads,
                env.getReporter());
    return new Parser(env.getRelativeWorkingDirectory(), RepositoryName.MAIN, repoMapping);
  }

  private QueryExpression createQueryExpression(List<String> targets)
      throws TargetFetcherException {
    String query = "deps(" + Joiner.on(" union ").join(targets) + ")";

    ImmutableMap<String, QueryFunction> functions =
        Stream.of(ConfiguredTargetQueryEnvironment.FUNCTIONS, env.getRuntime().getQueryFunctions())
            .flatMap(Collection::stream)
            .collect(toImmutableMap(QueryFunction::getName, Function.identity()));

    try {
      return QueryParser.parse(query, functions);
    } catch (QuerySyntaxException e) {
      throw new TargetFetcherException(
          String.format(
              "Fetching target dependencies for %s encountered an error: %s",
              QueryExpression.truncate(query), e.getMessage()));
    }
  }

  static void injectOptionsToFetchTarget(OptionsParser optionsParser) {
    try {
      optionsParser.parse(
          PriorityCategory.COMPUTED_DEFAULT,
          "Options required to fetch target",
          ImmutableList.of("--nobuild"));
      optionsParser.parse(
          PriorityCategory.COMPUTED_DEFAULT,
          "Fetch target should include 'tags = [\"manual\"]' targets by default",
          ImmutableList.of("--build_manual_tests"));
      optionsParser.parse(
          PriorityCategory.SOFTWARE_REQUIREMENT,
          "Fetch target should not exclude test_suite rules",
          ImmutableList.of("--noexpand_test_suites"));
      optionsParser.parse(
          PriorityCategory.SOFTWARE_REQUIREMENT,
          "Fetch target should not exclude tests",
          ImmutableList.of("--nobuild_tests_only"));
    } catch (OptionsParsingException e) {
      throw new IllegalStateException("Fetch target needed options failed to parse", e);
    }
  }

  static class TargetFetcherException extends Exception {
    public TargetFetcherException(String message) {
      super(message);
    }
  }
}
