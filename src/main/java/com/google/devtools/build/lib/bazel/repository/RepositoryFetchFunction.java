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

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.bzlmod.NonRegistryOverride;
import com.google.devtools.build.lib.bazel.bzlmod.VendorFileValue;
import com.google.devtools.build.lib.bazel.repository.RepositoryFunctionException.AlreadyReportedRepositoryAccessException;
import com.google.devtools.build.lib.bazel.repository.cache.LocalRepoContentsCache;
import com.google.devtools.build.lib.bazel.repository.cache.LocalRepoContentsCache.CandidateRepo;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.starlark.NeedsSkyframeRestartException;
import com.google.devtools.build.lib.bazel.repository.starlark.RepoMetadata;
import com.google.devtools.build.lib.bazel.repository.starlark.RepoMetadata.Reproducibility;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryContext;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryDefinitionLocationEvent;
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
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue.Success;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.runtime.RemoteRepoContentsCache;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.skyframe.AlreadyReportedException;
import com.google.devtools.build.lib.skyframe.IgnoredSubdirectoriesValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.RepoEnvironmentFunction;
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
import java.util.Optional;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
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

  private final BlazeDirectories directories;
  private final LocalRepoContentsCache repoContentsCache;
  private final Supplier<ImmutableMap<String, String>> repoEnvSupplier;
  private final Supplier<ImmutableMap<String, String>> nonstrictRepoEnvSupplier;

  private double timeoutScaling = 1.0;
  @Nullable private DownloadManager downloadManager;
  @Nullable private ProcessWrapper processWrapper = null;
  @Nullable private RepositoryRemoteExecutor repositoryRemoteExecutor;
  @Nullable private RemoteRepoContentsCache remoteRepoContentsCache;
  @Nullable private SyscallCache syscallCache;

  public RepositoryFetchFunction(
      Supplier<ImmutableMap<String, String>> repoEnvSupplier,
      Supplier<ImmutableMap<String, String>> nonstrictRepoEnvSupplier,
      BlazeDirectories directories,
      LocalRepoContentsCache repoContentsCache) {
    this.repoEnvSupplier = repoEnvSupplier;
    this.nonstrictRepoEnvSupplier = nonstrictRepoEnvSupplier;
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

  public void setRemoteRepoContentsCache(RemoteRepoContentsCache remoteRepoContentsCache) {
    this.remoteRepoContentsCache = remoteRepoContentsCache;
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

      RepoDefinition repoDefinition;
      switch ((RepoDefinitionValue) env.getValue(RepoDefinitionValue.key(repositoryName))) {
        case null -> {
          return null;
        }
        case RepoDefinitionValue.NotFound() -> {
          return new Failure(String.format("Repository '%s' is not defined", repositoryName));
        }
        case RepoDefinitionValue.RepoOverride(PathFragment repoPath) -> {
          return setupOverride(repoPath, env, repoRoot, repositoryName);
        }
        case RepoDefinitionValue.Found(RepoDefinition rd) -> {
          repoDefinition = rd;
        }
      }

      // See below (the `catch CancellationException` clause) for why there's a `while` loop here.
      while (true) {
        var state = env.getState(WorkerSkyKeyComputeState<RepositoryDirectoryValue>::new);
        try {
          return state.startOrContinueWork(
              env,
              "starlark-repository-" + repositoryName.getName(),
              (workerEnv) -> {
                return computeInternal(
                    workerEnv, repositoryName, starlarkSemantics, repoRoot, repoDefinition);
              });
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
                      repositoryName, "fetch interrupted due to memory pressure; restarting."));
        }
      }
    }
  }

  /**
   * The actual SkyFunction logic, run in a worker thread. Note that, although the worker thread
   * never sees Skyframe restarts, {@code env.valuesMissing()} can still be true due to deps in
   * error. So this function still needs to return {@code null} when appropriate. See Javadoc of
   * {@link com.google.devtools.build.skyframe.WorkerSkyFunctionEnvironment} for more information.
   */
  @Nullable
  private RepositoryDirectoryValue computeInternal(
      Environment env,
      RepositoryName repositoryName,
      StarlarkSemantics starlarkSemantics,
      Path repoRoot,
      RepoDefinition repoDefinition)
      throws InterruptedException, RepositoryFunctionException {
    var digestWriter =
        DigestWriter.create(env, directories, repositoryName, repoDefinition, starlarkSemantics);
    if (digestWriter == null) {
      return null;
    }

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

    if (shouldUseCachedRepoContents(env, repoDefinition)) {
      // Make sure marker file is up-to-date; correctly describes the current repository state
      if (digestWriter.areRepositoryAndMarkerFileConsistent(env).isEmpty()) {
        return new Success(Root.fromPath(repoRoot), excludeRepoFromVendoring);
      }
      if (env.valuesMissing()) {
        return null;
      }

      // Then check if the local repo contents cache has this.
      if (repoContentsCache.isEnabled()) {
        ImmutableList<CandidateRepo> candidateRepos =
            repoContentsCache.getCandidateRepos(digestWriter.predeclaredInputHash);
        for (CandidateRepo candidate : candidateRepos) {
          if (digestWriter
              .areRepositoryAndMarkerFileConsistent(env, candidate.recordedInputsFile())
              .isEmpty()) {
            if (setupOverride(candidate.contentsDir().asFragment(), env, repoRoot, repositoryName)
                == null) {
              return null;
            }
            candidate.touch();
            return new Success(Root.fromPath(repoRoot), excludeRepoFromVendoring);
          }
          if (env.valuesMissing()) {
            return null;
          }
        }
      }

      if (remoteRepoContentsCache != null) {
        try {
          if (remoteRepoContentsCache.lookupCache(
              repositoryName, repoRoot, digestWriter.predeclaredInputHash, env.getListener())) {
            return new Success(Root.fromPath(repoRoot), excludeRepoFromVendoring);
          }
        } catch (IOException e) {
          env.getListener()
              .handle(
                  Event.warn(
                      "Remote repo contents cache lookup failed for %s: %s"
                          .formatted(repositoryName, e.getMessage())));
        }
      }
    }

    /* At this point: This is a force fetch, a local repository, OR The repository cache is old or
    didn't exist. In any of those cases, we initiate the fetching process UNLESS this is offline
    mode (fetching is disabled) */
    if (!RepositoryDirectoryValue.FETCH_DISABLED.get(env)) {
      // Fetching a repository is a long-running operation that can easily be interrupted. If it
      // is and the marker file exists on disk, a new call of this method may treat this
      // repository as valid even though it is in an inconsistent state. Clear the marker file and
      // only recreate it after fetching is done to prevent this scenario.
      DigestWriter.clearMarkerFile(directories, repositoryName);
      FetchResult result = fetchAndHandleEvents(repoDefinition, repoRoot, env, repositoryName);
      if (result == null) {
        return null;
      }
      digestWriter.writeMarkerFile(result.recordedInputValues());
      if (result.reproducible() == Reproducibility.YES && !repoDefinition.repoRule().local()) {
        // This repo is eligible for the local and remote repo contents cache.
        // Replant symlinks before caching to convert absolute symlinks pointing to the
        // workspace or external root into relative paths, making the cached repo portable.
        Path externalRepoRoot = RepositoryUtils.getExternalRepositoryDirectory(directories);
        boolean safeForLocalCacheReuse;
        try {
          safeForLocalCacheReuse =
              RepositoryUtils.replantSymlinks(
                  repoRoot,
                  directories.getWorkspace(),
                  externalRepoRoot,
                  PathFragment.EMPTY_FRAGMENT);
        } catch (IOException e) {
          throw new RepositoryFunctionException(
              new IOException(
                  "error replanting symlinks in repo %s before caching: %s"
                      .formatted(repositoryName, e.getMessage()),
                  e),
              Transience.TRANSIENT);
        }
        if (remoteRepoContentsCache != null) {
          remoteRepoContentsCache.addToCache(
              repositoryName,
              repoRoot,
              digestWriter.markerPath,
              digestWriter.predeclaredInputHash,
              env.getListener());
        }
        if (safeForLocalCacheReuse && repoContentsCache.isEnabled()) {
          CandidateRepo newCacheEntry;
          try {
            newCacheEntry =
                repoContentsCache.moveToCache(
                    repoRoot, digestWriter.markerPath, digestWriter.predeclaredInputHash);
          } catch (IOException e) {
            throw new RepositoryFunctionException(
                new IOException(
                    "error moving repo %s into the repo contents cache: %s"
                        .formatted(repositoryName, e.getMessage()),
                    e),
                Transience.TRANSIENT);
          }
          Path cachedRepoDir = newCacheEntry.contentsDir();
          RootedPath cachedRepoDirRootedPath =
              RootedPath.toRootedPath(
                  Root.absoluteRoot(cachedRepoDir.getFileSystem()), cachedRepoDir);
          // Don't forget to register a FileStateValue on the cache repo dir, so that we know to
          // refetch if the cache entry gets GC'd from under us or the entire cache is deleted.
          //
          // Note that registering a FileValue dependency instead would lead to subtly incorrect
          // behavior when the repo contents cache directory is deleted between builds:
          // 1. We register a FileValue dependency on the cache entry.
          // 2. Before the next build, the repo contents cache directory is deleted.
          // 3. On the next build, FileSystemValueChecker invalidates the underlying
          //    FileStateValue, which in turn results in the FileValue and the current
          //    RepositoryDirectoryValue being marked as dirty.
          // 4. Skyframe visits the dirty nodes bottom up to check for actual changes. In
          //    particular, it reevaluates FileFunction before RepositoryFetchFunction and thus
          //    the FileValue of the repo contents cache directory is locked in as non-existent
          //    before RepositoryFetchFunction can recreate it.
          // 5. Any other SkyFunction that depends on the FileValue of a file in the repo (e.g.
          //    PackageFunction) will report that file as missing since the resolved path has a
          //    parent that is non-existent.
          // By using FileStateValue directly, which benefits from special logic built into
          // DirtinessCheckerUtils that recognizes the repo contents cache directories with
          // non-UUID names and prevents locking in their value during dirtiness checking, we
          // avoid 4. and thus the incorrect missing file errors in 5.
          if (env.getValue(FileStateValue.key(cachedRepoDirRootedPath)) == null) {
            return null;
          }
        }
      }
      return new Success(Root.fromPath(repoRoot), excludeRepoFromVendoring);
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

    return new Success(Root.fromPath(repoRoot), excludeRepoFromVendoring);
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

      Optional<String> vendoredRepoOutOfDateReason =
          digestWriter.areRepositoryAndMarkerFileConsistent(env, vendorMarker);
      if (env.valuesMissing()) {
        return null;
      }
      // If our repo is up-to-date, or this is an offline build (--nofetch), then the vendored repo
      // is used.
      if (vendoredRepoOutOfDateReason.isEmpty()
          || (!RepositoryDirectoryValue.IS_VENDOR_COMMAND.get(env)
              && RepositoryDirectoryValue.FETCH_DISABLED.get(env))) {
        if (vendoredRepoOutOfDateReason.isPresent()) {
          env.getListener()
              .handle(
                  Event.warn(
                      String.format(
                          "Vendored repository '%s' is out-of-date (%s) and fetching is disabled."
                              + " Run build without the '--nofetch' option or run"
                              + " the bazel vendor command to update it",
                          repositoryName.getName(), vendoredRepoOutOfDateReason.get())));
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
                        repositoryName.getName(), vendoredRepoOutOfDateReason.get())));
      }
    } else if (vendorFile.pinnedRepos().contains(repositoryName)) {
      throw new RepositoryFunctionException(
          new IOException(
              "Pinned repository "
                  + repositoryName.getName()
                  + " not found under the vendor directory"),
          Transience.PERSISTENT);
    } else if (RepositoryDirectoryValue.FETCH_DISABLED.get(env)) {
      // repo not vendored & fetching is disabled (--nofetch)
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
    /* If fetching is enabled & this is a local repo: do NOT use cache!
     * Local repository are generally fast and do not rely on non-local data, making caching them
     * across server instances impractical. */
    if (!RepositoryDirectoryValue.FETCH_DISABLED.get(env) && repoDefinition.repoRule().local()) {
      return false;
    }

    boolean forceFetchEnabled = !RepositoryDirectoryValue.FORCE_FETCH.get(env).isEmpty();
    boolean forceFetchConfigureEnabled =
        repoDefinition.repoRule().configure()
            && !RepositoryDirectoryValue.FORCE_FETCH_CONFIGURE.get(env).isEmpty();

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
      RepoDefinition repoDefinition, Path repoRoot, Environment env, RepositoryName repoName)
      throws InterruptedException, RepositoryFunctionException {
    env.getListener().post(RepositoryFetchProgress.ongoing(repoName, "starting"));

    FetchResult result;
    try {
      result = fetch(repoDefinition, repoRoot, env, repoName);
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

  /**
   * The result of the {@link #fetch} method.
   *
   * @param recordedInputValues Any recorded inputs (and their values) encountered during the fetch
   *     of the repo. Changes to these inputs will result in the repo being refetched in the future.
   * @param reproducible Whether the fetched repo contents are reproducible, hence cacheable.
   */
  private record FetchResult(
      ImmutableList<RepoRecordedInput.WithValue> recordedInputValues,
      Reproducibility reproducible) {}

  @Nullable
  private FetchResult fetch(
      RepoDefinition repoDefinition, Path outputDirectory, Environment env, RepositoryName repoName)
      throws RepositoryFunctionException, InterruptedException {
    setupRepoRoot(outputDirectory);

    String defInfo = RepositoryResolvedEvent.getRuleDefinitionInformation(repoDefinition);
    env.getListener()
        .post(new StarlarkRepositoryDefinitionLocationEvent(repoDefinition.name(), defInfo));

    StarlarkCallable function = repoDefinition.repoRule().impl();
    ImmutableMap<String, Optional<String>> envVarValues =
        RepoEnvironmentFunction.getEnvironmentView(env, repoDefinition.repoRule().environ());
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

    ImmutableList<RepoRecordedInput.WithValue> recordedInputValues;
    RepoMetadata repoMetadata;
    try (Mutability mu = Mutability.create("Starlark repository");
        StarlarkRepositoryContext starlarkRepositoryContext =
            new StarlarkRepositoryContext(
                repoDefinition,
                packageLocator,
                outputDirectory,
                ignoredSubdirectories.asIgnoredSubdirectories(),
                env,
                repoEnvSupplier.get(),
                ImmutableMap.copyOf(nonstrictRepoEnvSupplier.get()),
                downloadManager,
                timeoutScaling,
                processWrapper,
                starlarkSemantics,
                repositoryRemoteExecutor,
                syscallCache,
                directories)) {
      StarlarkThread thread =
          StarlarkThread.create(
              mu,
              starlarkSemantics,
              "repository " + repoName.getDisplayForm(mainRepoMapping),
              SymbolGenerator.create("fetching " + repoName));
      thread.setPrintHandler(Event.makeDebugPrintHandler(env.getListener()));
      starlarkRepositoryContext.storeRepoMappingRecorderInThread(thread);

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
          Profiler.instance().profile(ProfilerTask.STARLARK_REPOSITORY_FN, repoDefinition::name)) {
        result = Starlark.positionalOnlyCall(thread, function, starlarkRepositoryContext);
        starlarkRepositoryContext.markSuccessful();
      }

      repoMetadata =
          switch (result) {
            case Dict<?, ?> dict ->
                new RepoMetadata(
                    RepoMetadata.Reproducibility.NO,
                    Dict.cast(dict, String.class, Object.class, "return value"));
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

      recordedInputValues = starlarkRepositoryContext.getRecordedInputs();
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
    return new Success(Root.fromPath(source), /* excludeFromVendoring= */ true);
  }
}
