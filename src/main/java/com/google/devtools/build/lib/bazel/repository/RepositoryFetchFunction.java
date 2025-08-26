// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.repository;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.skyframe.RepositoryMappingFunction.REPOSITORY_OVERRIDES;

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.bzlmod.NonRegistryOverride;
import com.google.devtools.build.lib.bazel.bzlmod.VendorFileValue;
import com.google.devtools.build.lib.bazel.repository.RepositoryFunctionException.AlreadyReportedRepositoryAccessException;
import com.google.devtools.build.lib.bazel.repository.cache.RepoContentsCache;
import com.google.devtools.build.lib.bazel.repository.cache.RepoContentsCache.CandidateRepo;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.starlark.NeedsSkyframeRestartException;
import com.google.devtools.build.lib.bazel.repository.starlark.RepoMetadata;
import com.google.devtools.build.lib.bazel.repository.starlark.RepoMetadata.Reproducibility;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryContext;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryDefinitionLocationEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.StarlarkThreadContext;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.repository.RepositoryFailedEvent;
import com.google.devtools.build.lib.repository.RepositoryFetchProgress;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue.Failure;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.skyframe.AlreadyReportedException;
import com.google.devtools.build.lib.skyframe.IgnoredSubdirectoriesValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WorkerSkyKeyComputeState;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.SymbolGenerator;

/** A {@link SkyFunction} that fetches the given repository. */
public final class RepositoryFetchFunction implements SkyFunction {

  // This is a reference to isFetch in BazelRepositoryModule, which tracks whether the current
  // command is a fetch. Remote repository lookups are only allowed during fetches.
  private final AtomicBoolean isFetch;
  private final BlazeDirectories directories;
  private final RepoContentsCache repoContentsCache;
  private final Supplier<Map<String, String>> clientEnvironmentSupplier;

  private double timeoutScaling = 1.0;
  @Nullable private DownloadManager downloadManager;
  @Nullable private ProcessWrapper processWrapper = null;
  @Nullable private RepositoryRemoteExecutor repositoryRemoteExecutor;
  @Nullable private SyscallCache syscallCache;

  public RepositoryFetchFunction(
      Supplier<Map<String, String>> clientEnvironmentSupplier,
      AtomicBoolean isFetch,
      BlazeDirectories directories,
      RepoContentsCache repoContentsCache) {
    this.clientEnvironmentSupplier = clientEnvironmentSupplier;
    this.isFetch = isFetch;
    this.directories = directories;
    this.repoContentsCache = repoContentsCache;
  }

  public void setTimeoutScaling(double timeoutScaling) {
    this.timeoutScaling = timeoutScaling;
  }

  public void setDownloadManager(DownloadManager downloadManager) {
    this.downloadManager = downloadManager;
  }

  public void setProcessWrapper(@Nullable ProcessWrapper processWrapper) {
    this.processWrapper = processWrapper;
  }

  public void setSyscallCache(SyscallCache syscallCache) {
    this.syscallCache = checkNotNull(syscallCache);
  }

  public void setRepositoryRemoteExecutor(RepositoryRemoteExecutor repositoryRemoteExecutor) {
    this.repositoryRemoteExecutor = repositoryRemoteExecutor;
  }

  /**
   * The result of the {@link #fetch} method.
   *
   * @param recordedInputValues Any recorded inputs (and their values) encountered during the fetch
   *     of the repo. Changes to these inputs will result in the repo being refetched in the future.
   *     Not an ImmutableMap, because regrettably the values can be null sometimes.
   * @param reproducible Whether the fetched repo contents are reproducible, hence cacheable.
   */
  private record FetchResult(
      Map<? extends RepoRecordedInput, String> recordedInputValues, Reproducibility reproducible) {}

  private static class State extends WorkerSkyKeyComputeState<FetchResult> {
    @Nullable FetchResult result;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, RepositoryFunctionException {
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    if (!repositoryName.isVisible()) {
      return new Failure(
          String.format(
              "No repository visible as '@%s' from %s",
              repositoryName.getName(), repositoryName.getContextRepoDisplayString()));
    }

    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.REPOSITORY_FETCH, repositoryName.toString())) {
      Path repoRoot =
          RepositoryUtils.getExternalRepositoryDirectory(directories)
              .getRelative(repositoryName.getName());
      Map<RepositoryName, PathFragment> overrides = REPOSITORY_OVERRIDES.get(env);
      if (Preconditions.checkNotNull(overrides).containsKey(repositoryName)) {
        return setupOverride(overrides.get(repositoryName), env, repoRoot, repositoryName);
      }
      if (repositoryName.equals(RepositoryName.BAZEL_TOOLS)) {
        return setupOverride(
            directories.getEmbeddedBinariesRoot().getRelative("embedded_tools").asFragment(),
            env,
            repoRoot,
            repositoryName);
      }

      RepoDefinition repoDefinition;
      switch ((RepoDefinitionValue) env.getValue(RepoDefinitionValue.key(repositoryName))) {
        case null -> {
          return null;
        }
        case RepoDefinitionValue.NotFound() -> {
          return new Failure(String.format("Repository '%s' is not defined", repositoryName));
        }
        case RepoDefinitionValue.Found(RepoDefinition rd) -> {
          repoDefinition = rd;
        }
      }

      DigestWriter digestWriter =
          new DigestWriter(directories, repositoryName, repoDefinition, starlarkSemantics);

      boolean excludeRepoFromVendoring = true;
      if (RepositoryDirectoryValue.VENDOR_DIRECTORY.get(env).isPresent()) { // If vendor mode is on
        VendorFileValue vendorFile = (VendorFileValue) env.getValue(VendorFileValue.KEY);
        if (env.valuesMissing()) {
          return null;
        }
        boolean excludeRepoByDefault = isRepoExcludedFromVendoringByDefault(repoDefinition);
        if (!excludeRepoByDefault && !vendorFile.ignoredRepos().contains(repositoryName)) {
          RepositoryDirectoryValue repositoryDirectoryValue =
              tryGettingValueUsingVendoredRepo(
                  env, repoRoot, repositoryName, digestWriter, vendorFile);
          if (env.valuesMissing()) {
            return null;
          }
          if (repositoryDirectoryValue != null) {
            return repositoryDirectoryValue;
          }
        }
        excludeRepoFromVendoring =
            excludeRepoByDefault
                || vendorFile.ignoredRepos().contains(repositoryName)
                || vendorFile.pinnedRepos().contains(repositoryName);
      }

      String predeclaredInputHash =
          DigestWriter.computePredeclaredInputHash(repoDefinition, starlarkSemantics);

      if (shouldUseCachedRepoContents(env, repoDefinition)) {
        // Make sure marker file is up-to-date; correctly describes the current repository state
        var repoState = digestWriter.areRepositoryAndMarkerFileConsistent(env);
        if (repoState == null) {
          return null;
        }
        if (repoState instanceof DigestWriter.RepoDirectoryState.UpToDate) {
          return new RepositoryDirectoryValue.Success(
              repoRoot, /* isFetchingDelayed= */ false, excludeRepoFromVendoring);
        }

        // Then check if the global repo contents cache has this.
        if (repoContentsCache.isEnabled()) {
          for (CandidateRepo candidate :
              repoContentsCache.getCandidateRepos(predeclaredInputHash)) {
            repoState =
                digestWriter.areRepositoryAndMarkerFileConsistent(
                    env, candidate.recordedInputsFile());
            if (repoState == null) {
              return null;
            }
            if (repoState instanceof DigestWriter.RepoDirectoryState.UpToDate) {
              if (setupOverride(candidate.contentsDir().asFragment(), env, repoRoot, repositoryName)
                  == null) {
                return null;
              }
              candidate.touch();
              return new RepositoryDirectoryValue.Success(
                  repoRoot, /* isFetchingDelayed= */ false, excludeRepoFromVendoring);
            }
          }
        }
      }

      /* At this point: This is a force fetch, a local repository, OR The repository cache is old or
      didn't exist. In any of those cases, we initiate the fetching process UNLESS this is offline
      mode (fetching is disabled) */
      if (isFetch.get()) {
        // Fetching a repository is a long-running operation that can easily be interrupted. If it
        // is and the marker file exists on disk, a new call of this method may treat this
        // repository as valid even though it is in an inconsistent state. Clear the marker file and
        // only recreate it after fetching is done to prevent this scenario.
        DigestWriter.clearMarkerFile(directories, repositoryName);
        FetchResult result = fetchAndHandleEvents(repoDefinition, repoRoot, env, skyKey);
        if (result == null) {
          return null;
        }
        digestWriter.writeMarkerFile(result.recordedInputValues());
        if (repoContentsCache.isEnabled()
            && result.reproducible() == RepoMetadata.Reproducibility.YES
            && !repoDefinition.repoRule().local()) {
          // This repo is eligible for the repo contents cache.
          Path cachedRepoDir;
          try {
            cachedRepoDir =
                repoContentsCache.moveToCache(
                    repoRoot, digestWriter.markerPath, predeclaredInputHash);
          } catch (IOException e) {
            throw new RepositoryFunctionException(
                new IOException(
                    "error moving repo %s into the repo contents cache: %s"
                        .formatted(repositoryName, e.getMessage()),
                    e),
                Transience.TRANSIENT);
          }
          // Don't forget to register a FileValue on the cache repo dir, so that we know to refetch
          // if the cache entry gets GC'd from under us.
          if (env.getValue(
                  FileValue.key(
                      RootedPath.toRootedPath(
                          Root.absoluteRoot(cachedRepoDir.getFileSystem()), cachedRepoDir)))
              == null) {
            return null;
          }
        }
        return new RepositoryDirectoryValue.Success(
            repoRoot, /* isFetchingDelayed= */ false, excludeRepoFromVendoring);
      }

      if (!repoRoot.exists()) {
        // The repository isn't on the file system, there is nothing we can do.
        throw new RepositoryFunctionException(
            new IOException(
                "to fix, run\n\tbazel fetch //...\nExternal repository "
                    + repositoryName
                    + " not found and fetching repositories is disabled."),
            Transience.TRANSIENT);
      }

      // Try to build with whatever is on the file system and emit a warning.
      env.getListener()
          .handle(
              Event.warn(
                  String.format(
                      "External repository '%s' is not up-to-date and fetching is disabled. To"
                          + " update, run the build without the '--nofetch' command line option.",
                      repositoryName)));

      return new RepositoryDirectoryValue.Success(
          repoRoot, /* isFetchingDelayed= */ true, excludeRepoFromVendoring);
    }
  }

  @Nullable
  private RepositoryDirectoryValue tryGettingValueUsingVendoredRepo(
      Environment env,
      Path repoRoot,
      RepositoryName repositoryName,
      DigestWriter digestWriter,
      VendorFileValue vendorFile)
      throws RepositoryFunctionException, InterruptedException {
    Path vendorPath = RepositoryDirectoryValue.VENDOR_DIRECTORY.get(env).get();
    Path vendorRepoPath = vendorPath.getRelative(repositoryName.getName());
    if (vendorRepoPath.exists()) {
      Path vendorMarker = vendorPath.getChild(repositoryName.getMarkerFileName());
      if (vendorFile.pinnedRepos().contains(repositoryName)) {
        // pinned repos are used as they are without checking their marker file
        try {
          // delete the marker as it may become out-of-date while it's pinned (old version or
          // manual changes)
          vendorMarker.delete();
        } catch (IOException e) {
          throw new RepositoryFunctionException(e, Transience.TRANSIENT);
        }
        return setupOverride(vendorRepoPath.asFragment(), env, repoRoot, repositoryName);
      }

      DigestWriter.RepoDirectoryState vendoredRepoState =
          digestWriter.areRepositoryAndMarkerFileConsistent(env, vendorMarker);
      if (vendoredRepoState == null) {
        return null;
      }
      // If our repo is up-to-date, or this is an offline build (--nofetch), then the vendored repo
      // is used.
      if (vendoredRepoState instanceof DigestWriter.RepoDirectoryState.UpToDate
          || (!RepositoryDirectoryValue.IS_VENDOR_COMMAND.get(env).booleanValue()
              && !isFetch.get())) {
        if (vendoredRepoState instanceof DigestWriter.RepoDirectoryState.OutOfDate(String reason)) {
          env.getListener()
              .handle(
                  Event.warn(
                      String.format(
                          "Vendored repository '%s' is out-of-date (%s) and fetching is disabled."
                              + " Run build without the '--nofetch' option or run"
                              + " the bazel vendor command to update it",
                          repositoryName.getName(), reason)));
        }
        return setupOverride(vendorRepoPath.asFragment(), env, repoRoot, repositoryName);
      } else if (!RepositoryDirectoryValue.IS_VENDOR_COMMAND
          .get(env)
          .booleanValue()) { // build command & fetch enabled
        // We will continue fetching but warn the user that we are not using the vendored repo
        env.getListener()
            .handle(
                Event.warn(
                    String.format(
                        "Vendored repository '%s' is out-of-date (%s). The up-to-date version will"
                            + " be fetched into the external cache and used. To update the repo"
                            + " in the vendor directory, run the bazel vendor command",
                        repositoryName.getName(),
                        ((DigestWriter.RepoDirectoryState.OutOfDate) vendoredRepoState).reason())));
      }
    } else if (vendorFile.pinnedRepos().contains(repositoryName)) {
      throw new RepositoryFunctionException(
          new IOException(
              "Pinned repository "
                  + repositoryName.getName()
                  + " not found under the vendor directory"),
          Transience.PERSISTENT);
    } else if (!isFetch.get()) { // repo not vendored & fetching is disabled (--nofetch)
      throw new RepositoryFunctionException(
          new IOException(
              "Vendored repository "
                  + repositoryName.getName()
                  + " not found under the vendor directory and fetching is disabled."
                  + " To fix, run the bazel vendor command or build without the '--nofetch'"),
          Transience.TRANSIENT);
    }
    return null;
  }

  /**
   * Determines whether we should use cache repo contents (either the one in {@code
   * $outputBase/external} or any matching entry in the repo contents cache).
   */
  private boolean shouldUseCachedRepoContents(Environment env, RepoDefinition repoDefinition)
      throws InterruptedException {
    if (env.getState(State::new).result != null) {
      // If this SkyFunction has finished fetching once, then we should always use the cached
      // result. This means that we _very_ recently (as in, in the same command invocation) fetched
      // this repo (possibly with --force or --configure), and are only here again due to a Skyframe
      // restart very late into RepositoryDelegatorFunction.
      return true;
    }

    boolean forceFetchEnabled = !RepositoryDirectoryValue.FORCE_FETCH.get(env).isEmpty();
    boolean forceFetchConfigureEnabled =
        repoDefinition.repoRule().configure()
            && !RepositoryDirectoryValue.FORCE_FETCH_CONFIGURE.get(env).isEmpty();

    /* If fetching is enabled & this is a local repo: do NOT use cache!
     * Local repository are generally fast and do not rely on non-local data, making caching them
     * across server instances impractical. */
    if (isFetch.get() && repoDefinition.repoRule().local()) {
      return false;
    }

    /* For the non-local repositories, do NOT use cache if:
     * 1) Force fetch is enabled (bazel sync, or bazel fetch --force), OR
     * 2) Force fetch configure is enabled (bazel sync --configure) */
    if (forceFetchEnabled || forceFetchConfigureEnabled) {
      return false;
    }

    return true;
  }

  private boolean isRepoExcludedFromVendoringByDefault(RepoDefinition repoDefinition) {
    return repoDefinition.repoRule().local() || repoDefinition.repoRule().configure();
  }

  @Nullable
  private FetchResult fetchAndHandleEvents(
      RepoDefinition repoDefinition, Path repoRoot, Environment env, SkyKey skyKey)
      throws InterruptedException, RepositoryFunctionException {
    RepositoryName repoName = (RepositoryName) skyKey.argument();
    env.getListener().post(RepositoryFetchProgress.ongoing(repoName, "starting"));

    FetchResult result;
    try {
      result = fetch(repoDefinition, repoRoot, env, skyKey);
    } catch (RepositoryFunctionException e) {
      // Upon an exceptional exit, the fetching of that repository is over as well.
      env.getListener().post(RepositoryFetchProgress.finished(repoName));
      env.getListener().post(new RepositoryFailedEvent(repoName, e.getMessage()));

      if (e.getCause() instanceof AlreadyReportedException) {
        throw e;
      }
      env.getListener()
          .handle(
              Event.error(String.format("fetching %s: %s", repoDefinition.name(), e.getMessage())));

      // Rewrap the underlying exception to signal callers not to re-report this error.
      throw new RepositoryFunctionException(
          new AlreadyReportedRepositoryAccessException(e.getCause()),
          e.isTransient() ? Transience.TRANSIENT : Transience.PERSISTENT);
    }

    if (env.valuesMissing()) {
      return null;
    }
    env.getListener().post(RepositoryFetchProgress.finished(repoName));
    return Preconditions.checkNotNull(result);
  }

  private static void setupRepoRoot(Path repoRoot) throws RepositoryFunctionException {
    try {
      repoRoot.deleteTree();
      Preconditions.checkNotNull(repoRoot.getParentDirectory()).createDirectoryAndParents();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  @Nullable
  private FetchResult fetch(
      RepoDefinition repoDefinition, Path outputDirectory, Environment env, SkyKey key)
      throws RepositoryFunctionException, InterruptedException {
    // See below (the `catch CancellationException` clause) for why there's a `while` loop here.
    while (true) {
      var state = env.getState(State::new);
      if (state.result != null) {
        // Escape early if we've already finished fetching once. This can happen if
        // a Skyframe restart is triggered _after_ fetch() is finished.
        return state.result;
      }
      try {
        state.result =
            state.startOrContinueWork(
                env,
                "starlark-repository-" + repoDefinition.name(),
                (workerEnv) -> {
                  setupRepoRoot(outputDirectory);
                  return fetchInternal(repoDefinition, outputDirectory, workerEnv, key);
                });
        return state.result;
      } catch (ExecutionException e) {
        Throwables.throwIfInstanceOf(e.getCause(), RepositoryFunctionException.class);
        Throwables.throwIfInstanceOf(e.getCause(), InterruptedException.class);
        Throwables.throwIfUnchecked(e.getCause());
        throw new IllegalStateException(
            "unexpected exception type: " + e.getCause().getClass(), e.getCause());
      } catch (CancellationException e) {
        // This can only happen if the state object was invalidated due to memory pressure, in
        // which case we can simply reattempt the fetch. Show a message and continue into the next
        // `while` iteration.
        env.getListener()
            .post(
                RepositoryFetchProgress.ongoing(
                    RepositoryName.createUnvalidated(repoDefinition.name()),
                    "fetch interrupted due to memory pressure; restarting."));
      }
    }
  }

  @Nullable
  private FetchResult fetchInternal(
      RepoDefinition repoDefinition, Path outputDirectory, Environment env, SkyKey key)
      throws RepositoryFunctionException, InterruptedException {

    String defInfo = RepositoryResolvedEvent.getRuleDefinitionInformation(repoDefinition);
    env.getListener()
        .post(new StarlarkRepositoryDefinitionLocationEvent(repoDefinition.name(), defInfo));

    StarlarkCallable function = repoDefinition.repoRule().impl();
    ImmutableMap<String, Optional<String>> envVarValues =
        RepositoryUtils.getEnvVarValues(env, repoDefinition.repoRule().environ());
    if (envVarValues == null) {
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

    @Nullable RepositoryMapping mainRepoMapping;
    if (NonRegistryOverride.BOOTSTRAP_REPO_RULES.contains(repoDefinition.repoRule().id())) {
      // Avoid a cycle.
      mainRepoMapping = null;
    } else {
      var mainRepoMappingValue =
          (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
      if (mainRepoMappingValue == null) {
        return null;
      }
      mainRepoMapping = mainRepoMappingValue.repositoryMapping();
    }

    IgnoredSubdirectoriesValue ignoredSubdirectories =
        (IgnoredSubdirectoriesValue) env.getValue(IgnoredSubdirectoriesValue.key());
    if (env.valuesMissing()) {
      return null;
    }

    Map<RepoRecordedInput, String> recordedInputValues = new LinkedHashMap<>();
    RepoMetadata repoMetadata;
    try (Mutability mu = Mutability.create("Starlark repository");
        StarlarkRepositoryContext starlarkRepositoryContext =
            new StarlarkRepositoryContext(
                repoDefinition,
                packageLocator,
                outputDirectory,
                ignoredSubdirectories.asIgnoredSubdirectories(),
                env,
                ImmutableMap.copyOf(clientEnvironmentSupplier.get()),
                downloadManager,
                timeoutScaling,
                processWrapper,
                starlarkSemantics,
                repositoryRemoteExecutor,
                syscallCache,
                directories)) {
      StarlarkThread thread =
          StarlarkThread.create(
              mu, starlarkSemantics, /* contextDescription= */ "", SymbolGenerator.create(key));
      thread.setPrintHandler(Event.makeDebugPrintHandler(env.getListener()));
      var repoMappingRecorder = new Label.RepoMappingRecorder();
      repoMappingRecorder.mergeEntries(repoDefinition.repoRule().recordedRepoMappingEntries());
      thread.setThreadLocal(Label.RepoMappingRecorder.class, repoMappingRecorder);

      // We sort of want a starlark thread context here, but no extra info is needed. So we just
      // use an anonymous class.
      new StarlarkThreadContext(() -> mainRepoMapping) {}.storeInThread(thread);
      if (starlarkRepositoryContext.isRemotable()) {
        // If a rule is declared remotable then invalidate it if remote execution gets
        // enabled or disabled.
        PrecomputedValue.REMOTE_EXECUTION_ENABLED.get(env);
      }

      // This rule is mainly executed for its side effect. Nevertheless, the return value is
      // of importance, as it provides information on how the call has to be modified to be a
      // reproducible rule.
      //
      // Also we do a lot of stuff in there, maybe blocking operations and we should certainly make
      // it possible to return null and not block but it doesn't seem to be easy with Starlark
      // structure as it is.
      Object result;
      try (SilentCloseable c =
          Profiler.instance()
              .profile(ProfilerTask.STARLARK_REPOSITORY_FN, () -> repoDefinition.name())) {
        result = Starlark.positionalOnlyCall(thread, function, starlarkRepositoryContext);
        starlarkRepositoryContext.markSuccessful();
      }

      repoMetadata =
          switch (result) {
            case Dict<?, ?> dict -> new RepoMetadata(RepoMetadata.Reproducibility.NO, dict);
            case RepoMetadata rm -> rm;
            default -> RepoMetadata.NONREPRODUCIBLE;
          };
      RepositoryResolvedEvent resolved =
          new RepositoryResolvedEvent(repoDefinition, repoMetadata.attrsForReproducibility());
      if (resolved.isNewInformationReturned()) {
        // TODO: https://github.com/bazelbuild/bazel/issues/26511 - printing this information isn't
        //  super useful, as it's often not actionable. Figure out what to do instead.
        env.getListener().handle(Event.debug(resolved.getMessage()));
        env.getListener().handle(Event.debug(defInfo));
      }

      // Modify marker data to include the files/dirents/env vars used by the rule's implementation
      // function.
      recordedInputValues.putAll(
          Maps.transformValues(RepoRecordedInput.EnvVar.wrap(envVarValues), v -> v.orElse(null)));
      recordedInputValues.putAll(starlarkRepositoryContext.getRecordedFileInputs());
      recordedInputValues.putAll(starlarkRepositoryContext.getRecordedDirentsInputs());
      recordedInputValues.putAll(starlarkRepositoryContext.getRecordedDirTreeInputs());
      recordedInputValues.putAll(
          Maps.transformValues(
              starlarkRepositoryContext.getRecordedEnvVarInputs(), v -> v.orElse(null)));

      for (Table.Cell<RepositoryName, String, RepositoryName> repoMappings :
          repoMappingRecorder.recordedEntries().cellSet()) {
        recordedInputValues.put(
            new RepoRecordedInput.RecordedRepoMapping(
                repoMappings.getRowKey(), repoMappings.getColumnKey()),
            repoMappings.getValue().getName());
      }
    } catch (NeedsSkyframeRestartException e) {
      return null;
    } catch (EvalException e) {
      env.getListener()
          .handle(
              Event.error(
                  e.getInnermostLocation(),
                  "An error occurred during the fetch of repository '"
                      + repoDefinition.name()
                      + "':\n   "
                      + e.getMessageWithStack()));
      env.getListener()
          .handle(Event.info(RepositoryResolvedEvent.getRuleDefinitionInformation(repoDefinition)));

      throw new RepositoryFunctionException(
          new AlreadyReportedRepositoryAccessException(e), Transience.TRANSIENT);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    if (!outputDirectory.isDirectory()) {
      throw new RepositoryFunctionException(
          new IOException(repoDefinition.name() + " must create a directory"),
          Transience.TRANSIENT);
    }

    // Make sure the fetched repo has a boundary file.
    if (!RepositoryUtils.isValidRepoRoot(outputDirectory)) {
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
      } catch (IOException e) {
        throw new RepositoryFunctionException(e, Transience.TRANSIENT);
      }
    }

    return new FetchResult(recordedInputValues, repoMetadata.reproducible());
  }

  @Nullable
  private RepositoryDirectoryValue setupOverride(
      PathFragment sourcePath, Environment env, Path repoRoot, RepositoryName repoName)
      throws RepositoryFunctionException, InterruptedException {
    DigestWriter.clearMarkerFile(directories, repoName);
    return symlinkRepoRoot(
        directories,
        repoRoot,
        directories.getWorkspace().getRelative(sourcePath),
        repoName.getName(),
        env);
  }

  @Nullable
  public static RepositoryDirectoryValue symlinkRepoRoot(
      BlazeDirectories directories,
      Path source,
      Path destination,
      String userDefinedPath,
      Environment env)
      throws RepositoryFunctionException, InterruptedException {
    if (source.isDirectory(Symlinks.NOFOLLOW)) {
      try {
        source.deleteTree();
      } catch (IOException e) {
        throw new RepositoryFunctionException(e, Transience.TRANSIENT);
      }
    }
    try {
      FileSystemUtils.ensureSymbolicLink(source, destination);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              String.format(
                  "Could not create symlink to repository \"%s\" (absolute path: \"%s\"): %s",
                  userDefinedPath, destination, e.getMessage()),
              e),
          Transience.TRANSIENT);
    }

    // Check that the target directory exists and is a directory.
    // Note that we have to check `destination` and not `source` here, otherwise we'd have a
    // circular dependency between SkyValues.
    RootedPath targetDirRootedPath;
    if (destination.startsWith(directories.getInstallBase())) {
      // The install base only changes with the Bazel binary so it's acceptable not to add its
      // ancestors as Skyframe dependencies.
      targetDirRootedPath =
          RootedPath.toRootedPath(Root.fromPath(destination), PathFragment.EMPTY_FRAGMENT);
    } else {
      targetDirRootedPath =
          RootedPath.toRootedPath(Root.absoluteRoot(destination.getFileSystem()), destination);
    }

    FileValue targetDirValue;
    try {
      targetDirValue =
          (FileValue) env.getValueOrThrow(FileValue.key(targetDirRootedPath), IOException.class);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException("Could not access " + destination + ": " + e.getMessage()),
          Transience.PERSISTENT);
    }
    if (targetDirValue == null) {
      // TODO(bazel-team): If this returns null, we unnecessarily recreate the symlink above on the
      // second execution.
      return null;
    }

    if (!targetDirValue.isDirectory()) {
      throw new RepositoryFunctionException(
          new IOException(
              String.format(
                  "The repository's path is \"%s\" (absolute: \"%s\") "
                      + "but it does not exist or is not a directory.",
                  userDefinedPath, destination)),
          Transience.PERSISTENT);
    }

    // Check that the directory contains a repo boundary file.
    // Note that we need to do this here since we're not creating a repo boundary file ourselves,
    // but entrusting the entire contents of the repo root to this target directory.
    if (!RepositoryUtils.isValidRepoRoot(destination)) {
      throw new RepositoryFunctionException(
          new IOException("No MODULE.bazel, REPO.bazel, or WORKSPACE file found in " + destination),
          Transience.TRANSIENT);
    }
    return new RepositoryDirectoryValue.Success(
        source, /* isFetchingDelayed= */ false, /* excludeFromVendoring= */ true);
  }
}
