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
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.InvalidArgumentException;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue.RepositoryMappingResolutionException;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.List;
import net.starlark.java.eval.EvalException;

/** Fetches repositories for commands. */
final class RepositoryFetcher {

  private final CommandEnvironment env;
  private final LoadingPhaseThreadsOption threadsOption;

  private RepositoryFetcher(
      CommandEnvironment env,
      LoadingPhaseThreadsOption threadsOption) {
    this.env = env;
    this.threadsOption = threadsOption;
  }

  static ImmutableMap<RepositoryName, RepositoryDirectoryValue> fetchRepos(
      List<String> repos,
      CommandEnvironment env,
      LoadingPhaseThreadsOption threadsOption)
      throws RepositoryMappingResolutionException,
          InterruptedException,
          RepositoryFetcherException {
    return new RepositoryFetcher(env, threadsOption).fetchRepos(repos);
  }

  private ImmutableMap<RepositoryName, RepositoryDirectoryValue> fetchRepos(List<String> repos)
      throws InterruptedException,
          RepositoryFetcherException,
          RepositoryMappingResolutionException {
    ImmutableSet<RepositoryName> reposnames = collectRepositoryNames(repos);
    EvaluationResult<SkyValue> evaluationResult = evaluateFetch(reposnames);
    return reposnames.stream()
        .collect(
            toImmutableMap(
                repoName -> repoName,
                repoName ->
                    (RepositoryDirectoryValue)
                        evaluationResult.get(RepositoryDirectoryValue.key(repoName))));
  }

  private EvaluationResult<SkyValue> evaluateFetch(ImmutableSet<RepositoryName> reposnames)
      throws InterruptedException, RepositoryFetcherException {
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setParallelism(threadsOption.threads)
            .setEventHandler(env.getReporter())
            .build();
    ImmutableSet<SkyKey> repoDelegatorKeys =
        reposnames.stream().map(RepositoryDirectoryValue::key).collect(toImmutableSet());
    EvaluationResult<SkyValue> evaluationResult =
        env.getSkyframeExecutor().prepareAndGet(repoDelegatorKeys, evaluationContext);
    if (evaluationResult.hasError()) {
      Exception e = evaluationResult.getError().getException();
      throw new RepositoryFetcherException(
          e != null ? e.getMessage() : "Unexpected error during repository fetching.");
    }
    return evaluationResult;
  }

  private ImmutableSet<RepositoryName> collectRepositoryNames(List<String> repos)
      throws InterruptedException,
          RepositoryFetcherException,
          RepositoryMappingResolutionException {
    ImmutableSet.Builder<RepositoryName> reposnames = ImmutableSet.builder();
    for (String repo : repos) {
      try {
        reposnames.add(getRepositoryName(repo));
      } catch (LabelSyntaxException | EvalException | InvalidArgumentException e) {
        throw new RepositoryFetcherException("Invalid repo name: " + e.getMessage());
      }
    }
    return reposnames.build();
  }

  private RepositoryName getRepositoryName(String repoName)
      throws EvalException,
          InterruptedException,
          LabelSyntaxException,
          InvalidArgumentException,
          RepositoryMappingResolutionException {
    if (repoName.startsWith("@@")) { // canonical RepoName
      return RepositoryName.create(repoName.substring(2));
    } else if (repoName.startsWith("@")) { // apparent RepoName
      RepositoryName.validateUserProvidedRepoName(repoName.substring(1));
      RepositoryMapping repoMapping =
          env.getSkyframeExecutor()
              .getMainRepoMapping(
                  env.getOptions().getOptions(KeepGoingOption.class).keepGoing,
                  threadsOption.threads,
                  env.getReporter());
      return repoMapping.get(repoName.substring(1));
    } else {
      throw new InvalidArgumentException(
          "The repo value has to be either apparent '@repo' or canonical '@@repo' repo name");
    }
  }

  static class RepositoryFetcherException extends Exception {
    public RepositoryFetcherException(String message) {
      super(message);
    }
  }
}
