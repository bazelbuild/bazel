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

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern.Parser;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.common.UniverseScope;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QuerySyntaxException;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.runtime.commands.QueryCommand;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue.RepositoryMappingResolutionException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.util.EnumSet;
import java.util.List;

/** Fetches all repos needed for building a given set of targets. */
public class TargetFetcher {
  private final CommandEnvironment env;

  private TargetFetcher(CommandEnvironment env) {
    this.env = env;
  }

  /** Uses `deps` query to find and fetch all repos needed for these targets */
  public static void fetchTargets(
      CommandEnvironment env, OptionsParsingResult options, List<String> targets)
      throws RepositoryMappingResolutionException, InterruptedException, TargetFetcherException {
    new TargetFetcher(env).fetchTargets(options, targets);
  }

  private void fetchTargets(OptionsParsingResult options, List<String> targets)
      throws InterruptedException, TargetFetcherException, RepositoryMappingResolutionException {
    AbstractBlazeQueryEnvironment<Target> queryEnv = getQueryEnv(options);
    QueryExpression expr = createQueryExpression(targets, queryEnv);
    QueryEvalResult queryEvalResult;
    try {
      queryEvalResult =
          queryEnv.evaluateQuery(
              expr,
              new ThreadSafeOutputFormatterCallback<>() {
                @Override
                public void processOutput(Iterable<Target> partialResult) {}
              });
    } catch (IOException e) {
      // Should be impossible since our OutputFormatterCallback doesn't throw IOException.
      throw new IllegalStateException(e);
    } catch (QueryException e) {
      throw new TargetFetcherException(
          String.format(
              "Fetching target dependencies for %s encountered an error: %s",
              expr, e.getMessage()));
    }

    if (!queryEvalResult.getSuccess()) {
      throw new TargetFetcherException(
          String.format(
              "Fetching some target dependencies for %s failed, but --keep_going specified. "
                  + " Ignoring errors",
              expr));
    }
  }

  AbstractBlazeQueryEnvironment<Target> getQueryEnv(OptionsParsingResult options)
      throws RepositoryMappingResolutionException, InterruptedException {
    boolean keepGoing = options.getOptions(KeepGoingOption.class).keepGoing;
    LoadingPhaseThreadsOption threadsOption = options.getOptions(LoadingPhaseThreadsOption.class);
    RepositoryMapping repoMapping =
        env.getSkyframeExecutor()
            .getMainRepoMapping(keepGoing, threadsOption.threads, env.getReporter());
    TargetPattern.Parser targetParser =
        new Parser(env.getRelativeWorkingDirectory(), RepositoryName.MAIN, repoMapping);
    return QueryCommand.newQueryEnvironment(
        env,
        keepGoing,
        false,
        UniverseScope.EMPTY,
        threadsOption.threads,
        EnumSet.noneOf(Setting.class),
        /* useGraphlessQuery= */ true,
        targetParser,
        LabelPrinter.legacy());
  }

  private QueryExpression createQueryExpression(
      List<String> targets, AbstractBlazeQueryEnvironment<Target> queryEnv)
      throws TargetFetcherException {
    String query = "deps(" + Joiner.on(" union ").join(targets) + ")";
    try {
      return QueryExpression.parse(query, queryEnv);
    } catch (QuerySyntaxException e) {
      throw new TargetFetcherException(
          String.format(
              "Fetching target dependencies for %s encountered an error: %s",
              QueryExpression.truncate(query), e.getMessage()));
    }
  }

  static class TargetFetcherException extends Exception {
    public TargetFetcherException(String message) {
      super(message);
    }
  }
}
