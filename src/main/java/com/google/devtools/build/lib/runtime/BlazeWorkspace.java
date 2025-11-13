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
package com.google.devtools.build.lib.runtime;

import static com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils.profiledAndLogged;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Range;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.SubscriberExceptionHandler;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.actions.cache.CompactPersistentActionCache;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.memory.AllocationTracker;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.server.IdleTask;
import com.google.devtools.build.lib.server.IdleTaskException;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCachingServicesSupplier;
import com.google.devtools.build.lib.util.io.CommandExtensionReporter;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.protobuf.Any;
import java.io.IOException;
import java.time.Duration;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * This class represents a workspace, and contains operations and data related to it. In contrast,
 * the BlazeRuntime class represents the Blaze server, and contains operations and data that are
 * (supposed to be) independent of the workspace or the current command.
 *
 * <p>At this time, there is still a 1:1 relationship between the BlazeRuntime and the
 * BlazeWorkspace, but the introduction of this class is a step towards allowing 1:N relationships.
 */
public final class BlazeWorkspace {
  static final String DO_NOT_BUILD_FILE_NAME = "DO_NOT_BUILD_HERE";

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final BlazeRuntime runtime;
  private final SubscriberExceptionHandler eventBusExceptionHandler;
  private final WorkspaceStatusAction.Factory workspaceStatusActionFactory;
  private final BinTools binTools;
  @Nullable private final AllocationTracker allocationTracker;

  private final BlazeDirectories directories;
  private final SkyframeExecutor skyframeExecutor;
  private final SyscallCache syscallCache;
  private final QuiescingExecutorsImpl quiescingExecutors;
  @Nullable private final Supplier<ObjectCodecRegistry> analysisCodecRegistrySupplier;

  /**
   * Null only during tests; should be created by a BlazeModule#workspaceInit hook for regular
   * operations.
   */
  @Nullable
  private final RemoteAnalysisCachingServicesSupplier remoteAnalysisCachingServicesSupplier;

  /**
   * The action cache, or null if it hasn't been loaded yet.
   *
   * <p>Loaded lazily by the first build command that enables the action cache. Cleared by a clean
   * command or by a build command that disables the action cache. Trimmed and reloaded by the
   * garbage collection idle task.
   */
  @Nullable private ActionCache actionCache;

  /** The execution time range of the previous build command in this server, if any. */
  @Nullable private Range<Long> lastExecutionRange = null;

  private final String outputBaseFilesystemTypeName;
  private final boolean allowExternalRepositories;
  @Nullable private final PathPackageLocator virtualPackageLocator;

  /** An {@link IdleTask} to garbage collect the action cache. */
  @VisibleForTesting
  final class ActionCacheGarbageCollectorIdleTask implements IdleTask {
    private final Duration delay;
    private final float threshold;
    private final Duration maxAge;

    ActionCacheGarbageCollectorIdleTask(Duration delay, float threshold, Duration maxAge) {
      this.delay = delay;
      this.threshold = threshold;
      this.maxAge = maxAge;
    }

    @Override
    public String displayName() {
      return "Action cache garbage collector";
    }

    @Override
    public Duration delay() {
      return delay;
    }

    @VisibleForTesting
    public float getThreshold() {
      return threshold;
    }

    @VisibleForTesting
    public Duration getMaxAge() {
      return maxAge;
    }

    @Override
    public void run() throws IdleTaskException, InterruptedException {
      try {
        if (actionCache == null) {
          // Do not load the action cache just to garbage collect it.
          return;
        }
        // Note that this reads and writes to the field in the outer class.
        actionCache = actionCache.trim(threshold, maxAge);
      } catch (IOException e) {
        throw new IdleTaskException(e);
      }
    }
  }

  public BlazeWorkspace(
      BlazeRuntime runtime,
      BlazeDirectories directories,
      SkyframeExecutor skyframeExecutor,
      SubscriberExceptionHandler eventBusExceptionHandler,
      WorkspaceStatusAction.Factory workspaceStatusActionFactory,
      BinTools binTools,
      @Nullable AllocationTracker allocationTracker,
      SyscallCache syscallCache,
      Supplier<ObjectCodecRegistry> analysisCodecRegistrySupplier,
      @Nullable RemoteAnalysisCachingServicesSupplier remoteAnalysisCachingServicesSupplier,
      boolean allowExternalRepositories) {
    this.runtime = runtime;
    this.eventBusExceptionHandler = Preconditions.checkNotNull(eventBusExceptionHandler);
    this.workspaceStatusActionFactory = workspaceStatusActionFactory;
    this.binTools = binTools;
    this.allocationTracker = allocationTracker;

    this.directories = directories;
    this.skyframeExecutor = skyframeExecutor;
    this.syscallCache = syscallCache;
    this.quiescingExecutors = QuiescingExecutorsImpl.createDefault();
    this.allowExternalRepositories = allowExternalRepositories;
    this.virtualPackageLocator = createPackageLocatorIfVirtual(directories, skyframeExecutor);
    this.analysisCodecRegistrySupplier = analysisCodecRegistrySupplier;
    this.remoteAnalysisCachingServicesSupplier = remoteAnalysisCachingServicesSupplier;

    if (directories.inWorkspace()) {
      writeOutputBaseReadmeFile();
      writeDoNotBuildHereFile();
    }

    // Here we use outputBase instead of outputPath because we need a file system to create the
    // latter.
    this.outputBaseFilesystemTypeName = FileSystemUtils.getFileSystem(getOutputBase());
  }

  public BlazeRuntime getRuntime() {
    return runtime;
  }

  /**
   * Returns the Blaze directories object for this runtime.
   */
  public BlazeDirectories getDirectories() {
    return directories;
  }

  public SkyframeExecutor getSkyframeExecutor() {
    return skyframeExecutor;
  }

  public WorkspaceStatusAction.Factory getWorkspaceStatusActionFactory() {
    return workspaceStatusActionFactory;
  }

  public BinTools getBinTools() {
    return binTools;
  }

  /**
   * Returns the working directory of the server.
   *
   * <p>This is often the first entry on the {@code --package_path}, but not always.
   * Callers should certainly not make this assumption. The Path returned may be null.
   */
  public Path getWorkspace() {
    return directories.getWorkingDirectory();
  }

  /**
   * Returns the output base directory associated with this Blaze server
   * process. This is the base directory for shared Blaze state as well as tool
   * and strategy specific subdirectories.
   */
  public Path getOutputBase() {
    return directories.getOutputBase();
  }

  /**
   * Returns the cached value of {@code
   * getOutputBase().getFilesystem().getFileSystemType(getOutputBase())}, which is assumed to be
   * constant for a fixed workspace for the life of the Blaze server.
   */
  public String getOutputBaseFilesystemTypeName() {
    return outputBaseFilesystemTypeName;
  }

  public Path getInstallBase() {
    return directories.getInstallBase();
  }

  /**
   * Returns the path to the action cache directory.
   *
   * <p>This path must be a descendant of the output base, as the action cache cannot be safely
   * shared between different workspaces.
   */
  private Path getActionCacheDirectory() {
    return getOutputBase().getChild("action_cache");
  }

  /**
   * Returns the path where an action cache previously determined to be corrupted is stored. *
   *
   * <p>This path must be a descendant of the output base, as the action cache cannot be safely
   * shared between different workspaces.
   */
  private Path getCorruptedActionCacheDirectory() {
    return getOutputBase().getChild("action_cache.bad");
  }

  /**
   * Returns the path where the action cache may temporarily store data during garbage collection.
   *
   * <p>This path must be a descendant of the output base, as the action cache cannot be safely
   * shared between different workspaces.
   */
  private Path getActionCacheTmpDirectory() {
    return getOutputBase().getChild("action_cache.tmp");
  }

  /** Returns an {@link IdleTask} to garbage collect the action cache. */
  public IdleTask getActionCacheGcIdleTask(Duration delay, float threshold, Duration maxAge) {
    return new ActionCacheGarbageCollectorIdleTask(delay, threshold, maxAge);
  }

  void recordLastExecutionTime(long commandStartTime) {
    long currentTimeMillis = runtime.getClock().currentTimeMillis();
    lastExecutionRange =
        currentTimeMillis >= commandStartTime
            ? Range.closed(commandStartTime, currentTimeMillis)
            : null;
  }

  /**
   * Range that represents the last execution time of a build in millis since epoch.
   */
  @Nullable
  public Range<Long> getLastExecutionTimeRange() {
    return lastExecutionRange;
  }

  /**
   * Initializes a CommandEnvironment to execute a command in this workspace.
   *
   * <p>This method should be called from the "main" thread on which the command will execute; that
   * thread will receive interruptions if a module requests an early exit.
   *
   * @param warnings a list of warnings to which the CommandEnvironment can add any warning
   *     generated during initialization. This is needed because Blaze's output handling is not yet
   *     fully configured at this point.
   */
  public CommandEnvironment initCommand(
      Command command,
      OptionsParsingResult options,
      InvocationPolicy invocationPolicy,
      List<String> warnings,
      long waitTimeInMs,
      long commandStartTime,
      @Nullable ImmutableList<IdleTask.Result> idleTaskResultsFromPreviousIdlePeriod,
      Consumer<String> shutdownReasonConsumer,
      List<Any> commandExtensions,
      CommandExtensionReporter commandExtensionReporter,
      int attemptNumber,
      @Nullable String buildRequestIdOverride,
      ConfigFlagDefinitions configFlagDefinitions) {
    quiescingExecutors.resetParameters(options);
    CommandEnvironment env =
        new CommandEnvironment(
            runtime,
            this,
            new EventBus(eventBusExceptionHandler),
            Thread.currentThread(),
            command,
            options,
            invocationPolicy,
            getOrCreatePackageLocatorForCommand(options),
            syscallCache,
            quiescingExecutors,
            warnings,
            waitTimeInMs,
            commandStartTime,
            idleTaskResultsFromPreviousIdlePeriod,
            shutdownReasonConsumer,
            commandExtensions,
            commandExtensionReporter,
            attemptNumber,
            buildRequestIdOverride,
            configFlagDefinitions,
            new ResourceManager());
    skyframeExecutor.setClientEnv(env.getClientEnv());
    BuildRequestOptions buildRequestOptions = options.getOptions(BuildRequestOptions.class);
    if (buildRequestOptions != null && !buildRequestOptions.useActionCache) {
      // Drop the action cache reference to save memory since we don't need it for this build. If a
      // subsequent build needs it, getOrLoadPersistentActionCache will reload it from disk.
      actionCache = null;
    }
    return env;
  }

  void clearEventBus() {
    // EventBus does not have an unregister() method, so this is how we release memory associated
    // with handlers.
    skyframeExecutor.setEventBus(null);
  }

  /**
   * Reinitializes the Skyframe evaluator.
   */
  public void resetEvaluator() {
    skyframeExecutor.resetEvaluator();
  }

  /** Removes in-memory and on-disk action caches. */
  public void clearCaches() throws IOException {
    if (actionCache != null) {
      actionCache.clear();
    }
    actionCache = null;
    getActionCacheDirectory().deleteTree();
    getCorruptedActionCacheDirectory().deleteTree();
    getActionCacheTmpDirectory().deleteTree();
  }

  /**
   * Returns the action cache, loading it from disk if it isn't already loaded.
   *
   * <p>The returned reference is only valid for the current build request, as build options may
   * affect the presence of an action cache.
   */
  public ActionCache getOrLoadPersistentActionCache(Reporter reporter) throws IOException {
    if (actionCache == null) {
      try (AutoProfiler p = profiledAndLogged("Loading action cache", ProfilerTask.INFO)) {
        actionCache =
            CompactPersistentActionCache.create(
                getActionCacheDirectory(),
                getCorruptedActionCacheDirectory(),
                getActionCacheTmpDirectory(),
                runtime.getClock(),
                reporter);
      }
    }
    return actionCache;
  }

  /**
   * Returns the action cache, or null if it isn't already loaded.
   *
   * <p>The returned reference is only valid for the current build request, as build options may
   * affect the presence of an action cache.
   */
  @Nullable
  public ActionCache getPersistentActionCache() {
    return actionCache;
  }

  /**
   * Generates a README file in the output base directory. This README file
   * contains the name of the workspace directory, so that users can figure out
   * which output base directory corresponds to which workspace.
   */
  private void writeOutputBaseReadmeFile() {
    Preconditions.checkNotNull(getWorkspace());
    Path outputBaseReadmeFile = getOutputBase().getRelative("README");
    try {
      FileSystemUtils.writeIsoLatin1(
          outputBaseReadmeFile,
          "WORKSPACE: " + getWorkspace(),
          "",
          "The first line of this file is intentionally easy to parse for various",
          "interactive scripting and debugging purposes.  But please DO NOT write programs",
          "that exploit it, as they will be broken by design: it is not possible to",
          "reverse engineer the set of source trees or the --package_path from the output",
          "tree, and if you attempt it, you will fail, creating subtle and",
          "hard-to-diagnose bugs, that will no doubt get blamed on changes made by the",
          "Bazel team.",
          "",
          "This directory was generated by Bazel.",
          "Do not attempt to modify or delete any files in this directory.",
          "Among other issues, Bazel's file system caching assumes that",
          "only Bazel will modify this directory and the files in it,",
          "so if you change anything here you may mess up Bazel's cache.");
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Couldn't write to '%s'", outputBaseReadmeFile);
    }
  }

  private void writeDoNotBuildHereFile(Path filePath) {
    try {
      filePath.getParentDirectory().createDirectoryAndParents();
      FileSystemUtils.writeContent(filePath, ISO_8859_1, getWorkspace().toString());
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Couldn't write to '%s'", filePath);
    }
  }

  private void writeDoNotBuildHereFile() {
    Preconditions.checkNotNull(getWorkspace());
    writeDoNotBuildHereFile(getOutputBase().getRelative(DO_NOT_BUILD_FILE_NAME));
    writeDoNotBuildHereFile(
        getOutputBase().getRelative("execroot").getRelative(DO_NOT_BUILD_FILE_NAME));
  }

  @Nullable
  public AllocationTracker getAllocationTracker() {
    return allocationTracker;
  }

  public boolean doesAllowExternalRepositories() {
    return allowExternalRepositories;
  }

  @Nullable
  public Supplier<ObjectCodecRegistry> getAnalysisObjectCodecRegistrySupplier() {
    return analysisCodecRegistrySupplier;
  }

  public RemoteAnalysisCachingServicesSupplier remoteAnalysisCachingServicesSupplier() {
    return remoteAnalysisCachingServicesSupplier;
  }

  @Nullable
  private static PathPackageLocator createPackageLocatorIfVirtual(
      BlazeDirectories directories, SkyframeExecutor skyframeExecutor) {
    Root virtualSourceRoot = directories.getVirtualSourceRoot();
    if (virtualSourceRoot == null) {
      return null;
    }
    return PathPackageLocator.createWithoutExistenceCheck(
        /* outputBase= */ null,
        ImmutableList.of(virtualSourceRoot),
        skyframeExecutor.getBuildFilesByPriority());
  }

  @Nullable // Null for commands that don't have PackageOptions (version, help, shutdown, etc).
  private PathPackageLocator getOrCreatePackageLocatorForCommand(OptionsParsingResult options) {
    var packageOptions = options.getOptions(PackageOptions.class);
    Path workspace = directories.getWorkspace();
    if (packageOptions == null || workspace == null) {
      return null;
    }
    if (virtualPackageLocator != null) {
      return virtualPackageLocator;
    }
    return PathPackageLocator.create(
        directories.getOutputBase(),
        packageOptions.packagePath,
        NullEventHandler.INSTANCE,
        workspace.asFragment(),
        workspace,
        skyframeExecutor.getBuildFilesByPriority());
  }
}
