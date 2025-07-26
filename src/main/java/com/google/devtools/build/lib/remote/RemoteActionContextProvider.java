// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.exec.SpawnCache;
import com.google.devtools.build.lib.exec.SpawnStrategyRegistry;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.remote.common.RemoteExecutionClient;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.common.RemotePathResolver.DefaultRemotePathResolver;
import com.google.devtools.build.lib.remote.common.RemotePathResolver.SiblingRepositoryLayoutResolver;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.TempPathGenerator;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Set;
import java.util.concurrent.Executor;
import javax.annotation.Nullable;

/** Provides a remote execution context. */
final class RemoteActionContextProvider {

  private final Executor executor;
  private final CommandEnvironment env;
  @Nullable private final CombinedCache combinedCache;
  @Nullable private final RemoteExecutionClient remoteExecutor;
  @Nullable private final ListeningScheduledExecutorService retryScheduler;
  private final DigestUtil digestUtil;
  @Nullable private final Path logDir;
  private TempPathGenerator tempPathGenerator;
  private RemoteExecutionService remoteExecutionService;
  @Nullable private RemoteActionInputFetcher actionInputFetcher;
  @Nullable private final RemoteOutputChecker remoteOutputChecker;
  @Nullable private final OutputService outputService;
  private final Set<Digest> knownMissingCasDigests;

  private RemoteActionContextProvider(
      Executor executor,
      CommandEnvironment env,
      @Nullable CombinedCache combinedCache,
      @Nullable RemoteExecutionClient remoteExecutor,
      @Nullable ListeningScheduledExecutorService retryScheduler,
      DigestUtil digestUtil,
      @Nullable Path logDir,
      @Nullable RemoteOutputChecker remoteOutputChecker,
      @Nullable OutputService outputService,
      Set<Digest> knownMissingCasDigests) {
    this.executor = executor;
    this.env = Preconditions.checkNotNull(env, "env");
    this.combinedCache = combinedCache;
    this.remoteExecutor = remoteExecutor;
    this.retryScheduler = retryScheduler;
    this.digestUtil = digestUtil;
    this.logDir = logDir;
    this.remoteOutputChecker = remoteOutputChecker;
    this.outputService = outputService;
    this.knownMissingCasDigests = knownMissingCasDigests;
  }

  public static RemoteActionContextProvider createForPlaceholder(
      CommandEnvironment env,
      ListeningScheduledExecutorService retryScheduler,
      DigestUtil digestUtil,
      Set<Digest> knownMissingCasDigests) {
    return new RemoteActionContextProvider(
        directExecutor(),
        env,
        /* combinedCache= */ null,
        /* remoteExecutor= */ null,
        retryScheduler,
        digestUtil,
        /* logDir= */ null,
        /* remoteOutputChecker= */ null,
        /* outputService= */ null,
        knownMissingCasDigests);
  }

  public static RemoteActionContextProvider createForRemoteCaching(
      Executor executor,
      CommandEnvironment env,
      CombinedCache combinedCache,
      ListeningScheduledExecutorService retryScheduler,
      DigestUtil digestUtil,
      @Nullable RemoteOutputChecker remoteOutputChecker,
      OutputService outputService,
      Set<Digest> knownMissingCasDigests) {
    return new RemoteActionContextProvider(
        executor,
        env,
        combinedCache,
        /* remoteExecutor= */ null,
        retryScheduler,
        digestUtil,
        /* logDir= */ null,
        remoteOutputChecker,
        checkNotNull(outputService),
        knownMissingCasDigests);
  }

  public static RemoteActionContextProvider createForRemoteExecution(
      Executor executor,
      CommandEnvironment env,
      RemoteExecutionCache remoteCache,
      RemoteExecutionClient remoteExecutor,
      ListeningScheduledExecutorService retryScheduler,
      DigestUtil digestUtil,
      Path logDir,
      @Nullable RemoteOutputChecker remoteOutputChecker,
      OutputService outputService,
      Set<Digest> knownMissingCasDigests) {
    return new RemoteActionContextProvider(
        executor,
        env,
        remoteCache,
        remoteExecutor,
        retryScheduler,
        digestUtil,
        logDir,
        remoteOutputChecker,
        checkNotNull(outputService),
        knownMissingCasDigests);
  }

  private RemotePathResolver createRemotePathResolver() {
    Path execRoot = env.getExecRoot();
    BuildLanguageOptions buildLanguageOptions =
        env.getOptions().getOptions(BuildLanguageOptions.class);
    RemotePathResolver remotePathResolver;
    if (buildLanguageOptions != null && buildLanguageOptions.experimentalSiblingRepositoryLayout) {
      remotePathResolver = new SiblingRepositoryLayoutResolver(execRoot);
    } else {
      remotePathResolver = new DefaultRemotePathResolver(execRoot);
    }
    return remotePathResolver;
  }

  public void setActionInputFetcher(RemoteActionInputFetcher actionInputFetcher) {
    this.actionInputFetcher = actionInputFetcher;
  }

  private RemoteExecutionService getRemoteExecutionService() {
    if (remoteExecutionService == null) {
      Path workingDirectory = env.getWorkingDirectory();
      RemoteOptions remoteOptions = checkNotNull(env.getOptions().getOptions(RemoteOptions.class));
      Path captureCorruptedOutputsDir = null;
      if (remoteOptions.remoteCaptureCorruptedOutputs != null
          && !remoteOptions.remoteCaptureCorruptedOutputs.isEmpty()) {
        captureCorruptedOutputsDir =
            workingDirectory.getRelative(remoteOptions.remoteCaptureCorruptedOutputs);
      }

      boolean verboseFailures =
          checkNotNull(env.getOptions().getOptions(ExecutionOptions.class)).verboseFailures;
      remoteExecutionService =
          new RemoteExecutionService(
              executor,
              env.getReporter(),
              verboseFailures,
              env.getExecRoot(),
              createRemotePathResolver(),
              env.getBuildRequestId(),
              env.getCommandId().toString(),
              digestUtil,
              checkNotNull(env.getOptions().getOptions(RemoteOptions.class)),
              checkNotNull(env.getOptions().getOptions(ExecutionOptions.class)),
              combinedCache,
              remoteExecutor,
              tempPathGenerator,
              captureCorruptedOutputsDir,
              remoteOutputChecker,
              outputService,
              knownMissingCasDigests);
      env.getEventBus().register(remoteExecutionService);
    }

    return remoteExecutionService;
  }

  /**
   * Registers a remote spawn strategy if this instance was created with an executor, otherwise does
   * nothing.
   *
   * @param registryBuilder builder with which to register the strategy
   */
  public void registerRemoteSpawnStrategy(SpawnStrategyRegistry.Builder registryBuilder) {
    ExecutionOptions executionOptions =
        checkNotNull(env.getOptions().getOptions(ExecutionOptions.class));
    RemoteSpawnRunner spawnRunner =
        new RemoteSpawnRunner(
            checkNotNull(env.getOptions().getOptions(RemoteOptions.class)),
            executionOptions.verboseFailures,
            env.getReporter(),
            retryScheduler,
            logDir,
            getRemoteExecutionService(),
            digestUtil);
    registryBuilder.registerStrategy(
        new RemoteSpawnStrategy(spawnRunner, executionOptions), "remote");
  }

  /**
   * Registers a spawn cache action context
   *
   * @param registryBuilder builder with which to register the cache
   */
  public void registerSpawnCache(ModuleActionContextRegistry.Builder registryBuilder) {
    RemoteSpawnCache spawnCache =
        new RemoteSpawnCache(
            checkNotNull(env.getOptions().getOptions(RemoteOptions.class)),
            checkNotNull(env.getOptions().getOptions(ExecutionOptions.class)).verboseFailures,
            getRemoteExecutionService(),
            digestUtil);
    registryBuilder.register(SpawnCache.class, spawnCache, "remote-cache");
  }

  CombinedCache getCombinedCache() {
    return combinedCache;
  }

  RemoteExecutionClient getRemoteExecutionClient() {
    return remoteExecutor;
  }

  void setTempPathGenerator(TempPathGenerator tempPathGenerator) {
    this.tempPathGenerator = tempPathGenerator;
  }

  public void afterCommand() {
    // actionInputFetcher uses combinedCache to prefetch inputs, so it must be shut down first.
    if (actionInputFetcher != null) {
      actionInputFetcher.shutdown();
    }
    if (remoteExecutionService != null) {
      remoteExecutionService.shutdown();
    } else {
      if (combinedCache != null) {
        combinedCache.release();
      }
      if (remoteExecutor != null) {
        remoteExecutor.close();
      }
    }

    if (outputService instanceof BazelOutputService bazelOutputService) {
      bazelOutputService.shutdown();
    }
  }
}
