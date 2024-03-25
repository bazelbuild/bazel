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

package com.google.devtools.build.lib.bazel.repository.starlark;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Table;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.repository.RepositoryResolvedEvent;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.starlark.RepoFetchingSkyKeyComputeState.Signal;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.SymbolGenerator;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.repository.RepositoryFetchProgress;
import com.google.devtools.build.lib.rules.repository.NeedsSkyframeRestartException;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.WorkspaceFileHelper;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.skyframe.IgnoredPackagePrefixesValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;

/** A repository function to delegate work done by Starlark remote repositories. */
public final class StarlarkRepositoryFunction extends RepositoryFunction {
  private final DownloadManager downloadManager;
  private double timeoutScaling = 1.0;
  @Nullable private ExecutorService workerExecutorService = null;
  @Nullable private ProcessWrapper processWrapper = null;
  @Nullable private RepositoryRemoteExecutor repositoryRemoteExecutor;
  @Nullable private SyscallCache syscallCache;

  public StarlarkRepositoryFunction(DownloadManager downloadManager) {
    this.downloadManager = downloadManager;
  }

  public void setTimeoutScaling(double timeoutScaling) {
    this.timeoutScaling = timeoutScaling;
  }

  public void setProcessWrapper(@Nullable ProcessWrapper processWrapper) {
    this.processWrapper = processWrapper;
  }

  public void setSyscallCache(SyscallCache syscallCache) {
    this.syscallCache = checkNotNull(syscallCache);
  }

  public void setWorkerExecutorService(@Nullable ExecutorService workerExecutorService) {
    this.workerExecutorService = workerExecutorService;
  }

  @Override
  protected void setupRepoRootBeforeFetching(Path repoRoot) throws RepositoryFunctionException {
    // DON'T delete the repo root here if we're using a worker thread, since when this SkyFunction
    // restarts, fetching is still happening inside the worker thread.
    if (workerExecutorService == null) {
      setupRepoRoot(repoRoot);
    }
  }

  @Override
  public void reportSkyframeRestart(Environment env, RepositoryName repoName) {
    // DON'T report a "restarting." event if we're using a worker thread, since the actual fetch
    // function run by the worker thread never restarts.
    if (workerExecutorService == null) {
      super.reportSkyframeRestart(env, repoName);
    }
  }

  @Nullable
  @Override
  public RepositoryDirectoryValue.Builder fetch(
      Rule rule,
      Path outputDirectory,
      BlazeDirectories directories,
      Environment env,
      Map<RepoRecordedInput, String> recordedInputValues,
      SkyKey key)
      throws RepositoryFunctionException, InterruptedException {
    if (workerExecutorService == null
        || env.inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors()) {
      // Don't use the worker thread if we're in Skyframe error bubbling. For some reason, using a
      // worker thread during error bubbling very frequently causes deadlocks on Linux platforms.
      // The deadlock is rather elusive and this is just the immediate thing that seems to help.
      // Fortunately, no Skyframe restarts should happen during error bubbling anyway, so this
      // shouldn't be a performance concern. See https://github.com/bazelbuild/bazel/issues/21238
      // for more context.
      return fetchInternal(rule, outputDirectory, directories, env, recordedInputValues, key);
    }
    var state = env.getState(RepoFetchingSkyKeyComputeState::new);
    var workerFuture = state.workerFuture;
    if (workerFuture == null) {
      // No worker is running yet, which means we're just starting to fetch this repo. Start with a
      // clean slate, and create the worker.
      setupRepoRoot(outputDirectory);
      Environment workerEnv = new RepoFetchingWorkerSkyFunctionEnvironment(state, env);
      workerFuture =
          workerExecutorService.submit(
              () -> {
                try {
                  return fetchInternal(
                      rule,
                      outputDirectory,
                      directories,
                      workerEnv,
                      state.recordedInputValues,
                      key);
                } finally {
                  state.signalQueue.put(Signal.DONE);
                }
              });
      state.workerFuture = workerFuture;
    } else {
      // A worker is already running. This can only mean one thing -- we just had a Skyframe
      // restart, and need to send over a fresh Environment.
      state.delegateEnvQueue.put(env);
    }
    Signal signal;
    try {
      signal = state.signalQueue.take();
    } catch (InterruptedException e) {
      // This means that we caught a Ctrl-C. Make sure to close the state object to interrupt the
      // worker thread, wait for it to finish, and then propagate the InterruptedException.
      state.close();
      signal = Uninterruptibles.takeUninterruptibly(state.signalQueue);
      // The call to Uninterruptibles.takeUninterruptibly() above may set the thread interrupted
      // status if it suppressed an InterruptedException, so we clear it again.
      Thread.interrupted();
      throw new InterruptedException();
    }
    switch (signal) {
      case RESTART:
        return null;
      case DONE:
        try {
          RepositoryDirectoryValue.Builder result = workerFuture.get();
          recordedInputValues.putAll(state.recordedInputValues);
          return result;
        } catch (ExecutionException e) {
          Throwables.throwIfInstanceOf(e.getCause(), RepositoryFunctionException.class);
          Throwables.throwIfUnchecked(e.getCause());
          throw new IllegalStateException(
              "unexpected exception type: " + e.getClass(), e.getCause());
        } catch (CancellationException e) {
          // This can only happen if the state object was invalidated due to memory pressure, in
          // which case we can simply reattempt the fetch.
          env.getListener()
              .post(
                  RepositoryFetchProgress.ongoing(
                      RepositoryName.createUnvalidated(rule.getName()),
                      "fetch interrupted due to memory pressure; restarting."));
          return fetch(rule, outputDirectory, directories, env, recordedInputValues, key);
        }
    }
    // TODO(wyv): use a switch expression above instead and remove this.
    throw new IllegalStateException();
  }

  @Nullable
  private RepositoryDirectoryValue.Builder fetchInternal(
      Rule rule,
      Path outputDirectory,
      BlazeDirectories directories,
      Environment env,
      Map<RepoRecordedInput, String> recordedInputValues,
      SkyKey key)
      throws RepositoryFunctionException, InterruptedException {

    String defInfo = RepositoryResolvedEvent.getRuleDefinitionInformation(rule);
    env.getListener().post(new StarlarkRepositoryDefinitionLocationEvent(rule.getName(), defInfo));

    StarlarkCallable function = rule.getRuleClassObject().getConfiguredTargetFunction();
    if (declareEnvironmentDependencies(recordedInputValues, env, getEnviron(rule)) == null) {
      return null;
    }
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (env.valuesMissing()) {
      return null;
    }

    PathPackageLocator packageLocator = PrecomputedValue.PATH_PACKAGE_LOCATOR.get(env);
    if (env.valuesMissing()) {
      return null;
    }

    IgnoredPackagePrefixesValue ignoredPackagesValue =
        (IgnoredPackagePrefixesValue) env.getValue(IgnoredPackagePrefixesValue.key());
    if (env.valuesMissing()) {
      return null;
    }
    ImmutableSet<PathFragment> ignoredPatterns = checkNotNull(ignoredPackagesValue).getPatterns();

    try (Mutability mu = Mutability.create("Starlark repository")) {
      StarlarkThread thread = new StarlarkThread(mu, starlarkSemantics);
      thread.setPrintHandler(Event.makeDebugPrintHandler(env.getListener()));
      var repoMappingRecorder = new Label.RepoMappingRecorder();
      // For repos defined in Bzlmod, record any used repo mappings in the marker file.
      // Repos defined in WORKSPACE are impossible to verify given the chunked loading (we'd have to
      // record which chunk the repo mapping was used in, and ain't nobody got time for that).
      if (!isWorkspaceRepo(rule)) {
        repoMappingRecorder.mergeEntries(
            rule.getRuleClassObject().getRuleDefinitionEnvironmentRepoMappingEntries());
        thread.setThreadLocal(Label.RepoMappingRecorder.class, repoMappingRecorder);
      }

      new BazelStarlarkContext(
              BazelStarlarkContext.Phase.LOADING, // ("fetch")
              new SymbolGenerator<>(key))
          .storeInThread(thread);

      StarlarkRepositoryContext starlarkRepositoryContext =
          new StarlarkRepositoryContext(
              rule,
              packageLocator,
              outputDirectory,
              ignoredPatterns,
              env,
              ImmutableMap.copyOf(clientEnvironment),
              downloadManager,
              timeoutScaling,
              processWrapper,
              starlarkSemantics,
              repositoryRemoteExecutor,
              syscallCache,
              directories);

      if (starlarkRepositoryContext.isRemotable()) {
        // If a rule is declared remotable then invalidate it if remote execution gets
        // enabled or disabled.
        PrecomputedValue.REMOTE_EXECUTION_ENABLED.get(env);
      }

      // Since restarting a repository function can be really expensive, we first ensure that
      // all label-arguments can be resolved to paths.
      try {
        starlarkRepositoryContext.enforceLabelAttributes();
      } catch (NeedsSkyframeRestartException e) {
        // Missing values are expected; just restart before we actually start the rule
        return null;
      }

      // This rule is mainly executed for its side effect. Nevertheless, the return value is
      // of importance, as it provides information on how the call has to be modified to be a
      // reproducible rule.
      //
      // Also we do a lot of stuff in there, maybe blocking operations and we should certainly make
      // it possible to return null and not block but it doesn't seem to be easy with Starlark
      // structure as it is.
      Object result;
      boolean fetchSuccessful = false;
      try (SilentCloseable c =
          Profiler.instance()
              .profile(ProfilerTask.STARLARK_REPOSITORY_FN, () -> rule.getLabel().toString())) {
        result =
            Starlark.call(
                thread,
                function,
                /*args=*/ ImmutableList.of(starlarkRepositoryContext),
                /*kwargs=*/ ImmutableMap.of());
        fetchSuccessful = true;
      } finally {
        if (starlarkRepositoryContext.ensureNoPendingAsyncTasks(
            env.getListener(), fetchSuccessful)) {
          if (fetchSuccessful) {
            throw new RepositoryFunctionException(
                new EvalException(
                    "Pending asynchronous work after repository rule finished running"),
                Transience.PERSISTENT);
          }
        }
      }

      RepositoryResolvedEvent resolved =
          new RepositoryResolvedEvent(
              rule, starlarkRepositoryContext.getAttr(), outputDirectory, result);
      if (resolved.isNewInformationReturned()) {
        env.getListener().handle(Event.debug(resolved.getMessage()));
        env.getListener().handle(Event.debug(defInfo));
      }

      // Modify marker data to include the files/dirents used by the rule's implementation function.
      recordedInputValues.putAll(starlarkRepositoryContext.getRecordedFileInputs());
      recordedInputValues.putAll(starlarkRepositoryContext.getRecordedDirentsInputs());
      recordedInputValues.putAll(starlarkRepositoryContext.getRecordedDirTreeInputs());

      // Ditto for environment variables accessed via `getenv`.
      for (String envKey : starlarkRepositoryContext.getAccumulatedEnvKeys()) {
        recordedInputValues.put(
            new RepoRecordedInput.EnvVar(envKey), clientEnvironment.get(envKey));
      }

      for (Table.Cell<RepositoryName, String, RepositoryName> repoMappings :
          repoMappingRecorder.recordedEntries().cellSet()) {
        recordedInputValues.put(
            new RepoRecordedInput.RecordedRepoMapping(
                repoMappings.getRowKey(), repoMappings.getColumnKey()),
            repoMappings.getValue().getName());
      }

      env.getListener().post(resolved);
    } catch (NeedsSkyframeRestartException e) {
      // A dependency is missing, cleanup and returns null
      try {
        if (outputDirectory.exists()) {
          outputDirectory.deleteTree();
        }
      } catch (IOException e1) {
        throw new RepositoryFunctionException(e1, Transience.TRANSIENT);
      }
      return null;
    } catch (EvalException e) {
      env.getListener()
          .handle(
              Event.error(
                  "An error occurred during the fetch of repository '"
                      + rule.getName()
                      + "':\n   "
                      + e.getMessageWithStack()));
      env.getListener()
          .handle(Event.info(RepositoryResolvedEvent.getRuleDefinitionInformation(rule)));

      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    if (!outputDirectory.isDirectory()) {
      throw new RepositoryFunctionException(
          new IOException(rule + " must create a directory"), Transience.TRANSIENT);
    }

    // Make sure the fetched repo has a boundary file.
    if (!WorkspaceFileHelper.isValidRepoRoot(outputDirectory)) {
      if (outputDirectory.isSymbolicLink()) {
        // The created repo is actually just a symlink to somewhere else (think local_repository).
        // In this case, we shouldn't try to create the repo boundary file ourselves, but report an
        // error instead.
        throw new RepositoryFunctionException(
            new IOException(
                "No MODULE.bazel, REPO.bazel, or WORKSPACE file found in " + outputDirectory),
            Transience.TRANSIENT);
      }
      // Otherwise, we can just create an empty REPO.bazel file.
      try {
        FileSystemUtils.createEmptyFile(outputDirectory.getRelative(LabelConstants.REPO_FILE_NAME));
        if (starlarkSemantics.getBool(BuildLanguageOptions.ENABLE_WORKSPACE)) {
          FileSystemUtils.createEmptyFile(
              outputDirectory.getRelative(LabelConstants.WORKSPACE_FILE_NAME));
        }
      } catch (IOException e) {
        throw new RepositoryFunctionException(e, Transience.TRANSIENT);
      }
    }

    return RepositoryDirectoryValue.builder().setPath(outputDirectory);
  }

  @SuppressWarnings("unchecked")
  private static ImmutableSet<String> getEnviron(Rule rule) {
    return ImmutableSet.copyOf((Iterable<String>) rule.getAttr("$environ"));
  }

  @Override
  protected boolean isLocal(Rule rule) {
    return (Boolean) rule.getAttr("$local");
  }

  @Override
  protected boolean isConfigure(Rule rule) {
    return (Boolean) rule.getAttr("$configure");
  }

  /**
   * Static method to determine if for a starlark repository rule {@code isConfigure} holds true. It
   * also checks that the rule is indeed a Starlark rule so that this class is the appropriate
   * handler for the given rule. As, however, only Starklark rules can be configure rules, this
   * method can also be used as a universal check.
   */
  public static boolean isConfigureRule(Rule rule) {
    return rule.getRuleClassObject().isStarlark() && ((Boolean) rule.getAttr("$configure"));
  }

  @Nullable
  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return null; // unused so safe to return null
  }

  public void setRepositoryRemoteExecutor(RepositoryRemoteExecutor repositoryRemoteExecutor) {
    this.repositoryRemoteExecutor = repositoryRemoteExecutor;
  }
}
