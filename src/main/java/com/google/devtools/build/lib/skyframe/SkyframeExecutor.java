// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Supplier;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionExecutionStatusReporter;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionLogBufferPathGenerator;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.blaze.BlazeDirectories;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.TransitivePackageLoader;
import com.google.devtools.build.lib.skyframe.DiffAwarenessManager.ProcessableModifiedFileSet;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ActionCompletedReceiver;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ProgressSupplier;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.ResourceUsage;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.lib.view.BuildView.Options;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.TopLevelArtifactContext;
import com.google.devtools.build.lib.view.WorkspaceStatusAction;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory.BuildInfoKey;
import com.google.devtools.build.lib.view.config.BinTools;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.view.config.BuildConfigurationKey;
import com.google.devtools.build.lib.view.config.BuildOptions;
import com.google.devtools.build.lib.view.config.ConfigurationFactory;
import com.google.devtools.build.lib.view.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.view.config.InvalidConfigurationException;
import com.google.devtools.build.skyframe.BuildDriver;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.Differencer.Diff;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.ImmutableDiff;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator.EvaluatorSupplier;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * A helper object to support Skyframe-driven execution.
 *
 * <p>This object is mostly used to inject external state, such as the executor engine or
 * some additional artifacts (workspace status and build info artifacts) into SkyFunctions
 * for use during the build.
 */
public final class SkyframeExecutor {
  private final EvaluatorSupplier evaluatorSupplier;
  private MemoizingEvaluator memoizingEvaluator;
  private final MemoizingEvaluator.EmittedEventState emittedEventState =
      new MemoizingEvaluator.EmittedEventState();
  private final Reporter reporter;
  private final PackageFactory pkgFactory;
  private final WorkspaceStatusAction.Factory workspaceStatusActionFactory;
  private final BlazeDirectories directories;
  @Nullable
  private BatchStat batchStatter;

  // TODO(bazel-team): Figure out how to handle value builders that block internally. Blocking
  // operations may need to be handled in another (bigger?) thread pool. Also, we should detect
  // the number of cores and use that as the thread-pool size for CPU-bound operations.
  // I just bumped this to 200 to get reasonable execution phase performance; that may cause
  // significant overhead for CPU-bound processes (i.e. analysis). [skyframe-analysis]
  @VisibleForTesting
  public static final int DEFAULT_THREAD_COUNT = 200;

  private static final Logger LOG = Logger.getLogger(SkyframeExecutor.class.getName());

  // Stores Packages between reruns of the PackageFunction (because of missing dependencies,
  // within the same evaluate() run) to avoid loading the same package twice (first time loading
  // to find subincludes and declare value dependencies).
  // TODO(bazel-team): remove this cache once we have skyframe-native package loading
  // [skyframe-loading]
  private final ConcurrentMap<String, Package.LegacyBuilder> packageFunctionCache =
      Maps.newConcurrentMap();
  private final AtomicInteger numPackagesLoaded = new AtomicInteger(0);

  private SkyframeBuildView skyframeBuildView;
  private EventHandler errorEventListener;
  private ActionLogBufferPathGenerator actionLogBufferPathGenerator;

  private RecordingDifferencer recordingDiffer;
  private BuildDriver buildDriver;

  private final DiffAwarenessManager diffAwarenessManager;

  // AtomicReferences are used here as mutable boxes shared with value builders.
  private final AtomicBoolean showLoadingProgress = new AtomicBoolean();
  private final AtomicReference<UnixGlob.FilesystemCalls> syscalls =
      new AtomicReference<>(UnixGlob.DEFAULT_SYSCALLS);
  private final AtomicReference<PathPackageLocator> pkgLocator =
      new AtomicReference<>();
  private final AtomicReference<ImmutableSet<String>> deletedPackages =
      new AtomicReference<>(ImmutableSet.<String>of());
  private final AtomicReference<EventBus> eventBus = new AtomicReference<>();

  private final ImmutableList<BuildInfoFactory> buildInfoFactories;
  // Under normal circumstances, the artifact factory persists for the life of a Blaze server, but
  // since it is not yet created when we create the value builders, we have to use a supplier,
  // initialized when the build view is created.
  private final MutableSupplier<ArtifactFactory> artifactFactory = new MutableSupplier<>();
  // Used to give to WriteBuildInfoAction via a supplier. Relying on BuildVariableValue.BUILD_ID
  // would be preferable, but we have no way to have the Action depend on that value directly.
  // Having the BuildInfoFunction own the supplier is currently not possible either, because then
  // it would be invalidated on every build, since it would depend on the build id value.
  private MutableSupplier<UUID> buildId = new MutableSupplier<>();

  private boolean active = true;
  private boolean lastAnalysisDiscarded = false;
  private final PackageManager packageManager;

  private final Preprocessor.Factory.Supplier preprocessorFactorySupplier;
  private Preprocessor.Factory preprocessorFactory;

  /** Lower limit for size of {@link #allLoadedPackages} to consider clearing CT values. */
  private int valueCacheEvictionLimit = -1;

  /** Union of labels of loaded packages since the last eviction of CT values. */
  private Set<PathFragment> allLoadedPackages = ImmutableSet.of();

  // Use skyframe for execution? Alternative is to use legacy execution codepath.
  // TODO(bazel-team): Remove when legacy codepath is no longer used. [skyframe-execution]
  private final boolean skyframeBuild;

  // Can only be set once (to false) over the lifetime of this object. If false, the graph will not
  // store edges, saving memory but making incremental builds impossible.
  private boolean keepGraphEdges = true;

  private final TimestampGranularityMonitor tsgm;

  private final ResourceManager resourceManager;

  /** Used to lock evaluator on legacy calls to get existing values. */
  private final Object valueLookupLock = new Object();
  private final AtomicReference<ActionExecutionStatusReporter> statusReporterRef =
      new AtomicReference<>();
  private SkyframeActionExecutor skyframeActionExecutor;
  private SkyframeExecutor.SkyframeProgressReceiver progressReceiver;
  private AtomicReference<CyclesReporter> cyclesReporter = new AtomicReference<>();

  private BinTools binTools = null;
  private boolean needToInjectEmbeddedArtifacts = true;
  private int modifiedFiles;
  private final Predicate<PathFragment> allowedMissingInputs;

  private SkyframeIncrementalBuildMonitor incrementalBuildMonitor =
      new SkyframeIncrementalBuildMonitor();

  private MutableSupplier<ConfigurationFactory> configurationFactory = new MutableSupplier<>();
  private MutableSupplier<Map<String, String>> clientEnv = new MutableSupplier<>();
  private MutableSupplier<ImmutableList<ConfigurationFragmentFactory>> configurationFragments =
      new MutableSupplier<>();
  private SkyKey configurationSkyKey = null;

  @VisibleForTesting
  public SkyframeExecutor(Reporter reporter, PackageFactory pkgFactory,
                          TimestampGranularityMonitor tsgm, BlazeDirectories directories,
                          WorkspaceStatusAction.Factory workspaceStatusActionFactory,
                          ImmutableList<BuildInfoFactory> buildInfoFactories,
                          Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories) {
    this(reporter, pkgFactory, true, tsgm, directories, workspaceStatusActionFactory,
        buildInfoFactories, diffAwarenessFactories, Predicates.<PathFragment>alwaysFalse(),
        Preprocessor.Factory.Supplier.NullSupplier.INSTANCE);
  }

  public SkyframeExecutor(
      Reporter reporter,
      PackageFactory pkgFactory,
      boolean skyframeBuild,
      TimestampGranularityMonitor tsgm,
      BlazeDirectories directories,
      WorkspaceStatusAction.Factory workspaceStatusActionFactory,
      ImmutableList<BuildInfoFactory> buildInfoFactories,
      Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories,
      Predicate<PathFragment> allowedMissingInputs,
      Preprocessor.Factory.Supplier preprocessorFactorySupplier) {
    this(reporter, InMemoryMemoizingEvaluator.SUPPLIER, pkgFactory, skyframeBuild, tsgm,
        directories, workspaceStatusActionFactory, buildInfoFactories, diffAwarenessFactories,
        allowedMissingInputs, preprocessorFactorySupplier);
  }

  public SkyframeExecutor(
      Reporter reporter,
      EvaluatorSupplier evaluatorSupplier,
      PackageFactory pkgFactory,
      boolean skyframeBuild,
      TimestampGranularityMonitor tsgm,
      BlazeDirectories directories,
      WorkspaceStatusAction.Factory workspaceStatusActionFactory,
      ImmutableList<BuildInfoFactory> buildInfoFactories,
      Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories,
      Predicate<PathFragment> allowedMissingInputs,
      Preprocessor.Factory.Supplier preprocessorFactorySupplier) {
    // Strictly speaking, these arguments are not required for initialization, but all current
    // callsites have them at hand, so we might as well set them during construction.
    this.reporter = Preconditions.checkNotNull(reporter);
    this.evaluatorSupplier = evaluatorSupplier;
    this.pkgFactory = pkgFactory;
    this.pkgFactory.setSyscalls(syscalls);
    this.tsgm = tsgm;
    this.workspaceStatusActionFactory = workspaceStatusActionFactory;
    this.packageManager = new SkyframePackageManager(
        new SkyframePackageLoader(), new SkyframeTransitivePackageLoader(),
        new SkyframeTargetPatternEvaluator(this), syscalls, cyclesReporter, pkgLocator,
        numPackagesLoaded, this);
    this.skyframeBuild = skyframeBuild;
    this.errorEventListener = this.reporter;
    this.resourceManager = ResourceManager.instance();
    this.skyframeActionExecutor = new SkyframeActionExecutor(reporter, resourceManager, eventBus,
        statusReporterRef);
    this.directories = Preconditions.checkNotNull(directories);
    this.buildInfoFactories = buildInfoFactories;
    this.diffAwarenessManager = new DiffAwarenessManager(diffAwarenessFactories, reporter);
    this.allowedMissingInputs = allowedMissingInputs;
    this.preprocessorFactorySupplier = preprocessorFactorySupplier;
    this.preprocessorFactory = preprocessorFactorySupplier.getFactory(packageManager);
    pkgFactory.setPreprocessorFactory(preprocessorFactory);
    resetEvaluator();
  }

  private ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions(
      Root buildDataDirectory,
      PackageFactory pkgFactory,
      Predicate<PathFragment> allowedMissingInputs) {
    Map<SkyFunctionName, SkyFunction> map = new HashMap<>();
    map.put(SkyFunctions.BUILD_VARIABLE, new BuildVariableFunction());
    map.put(SkyFunctions.FILE_STATE, new FileStateFunction(tsgm, pkgLocator));
    map.put(SkyFunctions.DIRECTORY_LISTING_STATE, new DirectoryListingStateFunction(pkgLocator));
    map.put(SkyFunctions.FILE_SYMLINK_CYCLE_UNIQUENESS,
        new FileSymlinkCycleUniquenessFunction());
    map.put(SkyFunctions.FILE, new FileFunction(pkgLocator));
    map.put(SkyFunctions.DIRECTORY_LISTING, new DirectoryListingFunction());
    map.put(SkyFunctions.PACKAGE_LOOKUP, new PackageLookupFunction(pkgLocator, deletedPackages));
    map.put(SkyFunctions.CONTAINING_PACKAGE_LOOKUP, new ContainingPackageLookupFunction());
    map.put(SkyFunctions.AST_LOOKUP, new ASTFileLookupFunction(
        pkgLocator, packageManager, pkgFactory.getRuleClassProvider()));
    map.put(SkyFunctions.SKYLARK_IMPORTS_LOOKUP, new SkylarkImportLookupFunction(
        pkgFactory.getRuleClassProvider(), pkgFactory));
    map.put(SkyFunctions.GLOB, new GlobFunction());
    map.put(SkyFunctions.TARGET_PATTERN, new TargetPatternFunction(pkgLocator));
    map.put(SkyFunctions.RECURSIVE_PKG, new RecursivePkgFunction());
    map.put(SkyFunctions.PACKAGE, new PackageFunction(
        reporter, pkgFactory, packageManager, showLoadingProgress, packageFunctionCache,
        eventBus, numPackagesLoaded));
    map.put(SkyFunctions.TARGET_MARKER, new TargetMarkerFunction());
    map.put(SkyFunctions.TRANSITIVE_TARGET, new TransitiveTargetFunction());
    map.put(SkyFunctions.CONFIGURED_TARGET,
        new ConfiguredTargetFunction(new BuildViewProvider(), skyframeBuild));
    map.put(SkyFunctions.POST_CONFIGURED_TARGET,
        new PostConfiguredTargetFunction(new BuildViewProvider()));
    map.put(SkyFunctions.CONFIGURATION_COLLECTION, new ConfigurationCollectionFunction(
        configurationFactory, clientEnv));
    map.put(SkyFunctions.CONFIGURATION_FRAGMENT, new ConfigurationFragmentFunction(
        configurationFragments));
    map.put(SkyFunctions.WORKSPACE_FILE, new WorkspaceFileFunction());
    map.put(SkyFunctions.LABEL_BINDING, new LabelBindingFunction(pkgLocator));
    if (skyframeBuild) {
      map.put(SkyFunctions.TARGET_COMPLETION,
          new TargetCompletionFunction(new BuildViewProvider()));
      map.put(SkyFunctions.ARTIFACT, new ArtifactFunction(eventBus, allowedMissingInputs));
      map.put(SkyFunctions.BUILD_INFO_COLLECTION, new BuildInfoCollectionFunction(artifactFactory,
          buildDataDirectory));
      map.put(SkyFunctions.BUILD_INFO, new WorkspaceStatusFunction());
      map.put(SkyFunctions.ACTION_EXECUTION,
          new ActionExecutionFunction(skyframeActionExecutor, tsgm));
    }
    return ImmutableMap.copyOf(map);
  }

  @ThreadCompatible
  public void setActive(boolean active) {
    this.active = active;
  }

  private void checkActive() {
    Preconditions.checkState(active);
  }

  /**
   * If true, use Skyframe for execution phase. Alternative is to use legacy execution codepath.
   * TODO(bazel-team): Remove this when legacy execution is no longer used. [skyframe-execution]
   */
  public boolean skyframeBuild() {
    return skyframeBuild;
  }

  public void setFileCache(ActionInputFileCache fileCache) {
    this.skyframeActionExecutor.setFileCache(fileCache);
  }

  public void dump(PrintStream out) {
    memoizingEvaluator.dump(out);
  }

  public void dumpPackages(PrintStream out) {
    Iterable<SkyKey> packageSkyKeys = Iterables.filter(memoizingEvaluator.getValues().keySet(),
        SkyFunctions.isSkyFunction(SkyFunctions.PACKAGE));
    out.println(Iterables.size(packageSkyKeys) + " packages");
    for (SkyKey packageSkyKey : packageSkyKeys) {
      Package pkg = ((PackageValue) memoizingEvaluator.getValues().get(packageSkyKey)).getPackage();
      pkg.dump(out);
    }
  }

  public void setBatchStatter(@Nullable BatchStat batchStatter) {
    this.batchStatter = batchStatter;
  }

  /**
   * Notify listeners about changed files, and release any associated memory afterwards.
   */
  public void drainChangedFiles() {
    incrementalBuildMonitor.alertListeners(getEventBus());
    incrementalBuildMonitor = null;
  }

  @VisibleForTesting
  public BuildDriver getDriverForTesting() {
    return buildDriver;
  }

  class BuildViewProvider {
    /**
     * Returns the current {@link SkyframeBuildView} instance.
     */
    SkyframeBuildView getSkyframeBuildView() {
      return skyframeBuildView;
    }
  }

  /**
   * Reinitializes the Skyframe evaluator, dropping all previously computed values.
   *
   * <p>Be careful with this method as it also deletes all injected values. You need to make sure
   * that any necessary build variables are reinjected before the next build. Constants can be put
   * in {@link #reinjectConstantValues}.
   */
  @ThreadCompatible
  public void resetEvaluator() {
    emittedEventState.clear();
    recordingDiffer = new RecordingDifferencer();
    progressReceiver = new SkyframeProgressReceiver();
    Map<SkyFunctionName, SkyFunction> skyFunctions = skyFunctions(
        directories.getBuildDataDirectory(), pkgFactory, allowedMissingInputs);
    memoizingEvaluator = evaluatorSupplier.create(
        skyFunctions, recordingDiffer, progressReceiver, emittedEventState, keepGraphEdges);
    buildDriver = new SequentialBuildDriver(memoizingEvaluator);
    diffAwarenessManager.reset();
    if (skyframeBuildView != null) {
      skyframeBuildView.clearLegacyData();
    }
    reinjectConstantValues();
  }

  /**
   * Values whose values are known at startup and guaranteed constant are still wiped from the
   * evaluator when we create a new one, so they must be re-injected each time we create a new
   * evaluator.
   */
  private void reinjectConstantValues() {
    injectBuildInfoFactories();
    needToInjectEmbeddedArtifacts = true;
  }

  /**
   * Deletes all ConfiguredTarget values from the Skyframe cache. This is done to save memory (e.g.
   * on a configuration change); since the configuration is part of the key, these key/value pairs
   * will be sitting around doing nothing until the configuration changes back to the previous
   * value.
   *
   * <p>The next evaluation will delete all invalid values.
   */
  public void dropConfiguredTargets() {
    if (skyframeBuildView != null) {
      skyframeBuildView.clearInvalidatedConfiguredTargets();
    }
    memoizingEvaluator.delete(
        // We delete any value that can hold an action -- all subclasses of ActionLookupValue -- as
        // well as ActionExecutionValues, since they do not depend on ActionLookupValues.
        SkyFunctionName.functionIsIn(ImmutableSet.of(
            SkyFunctions.CONFIGURED_TARGET,
            SkyFunctions.ACTION_LOOKUP,
            SkyFunctions.BUILD_INFO,
            SkyFunctions.TARGET_COMPLETION,
            SkyFunctions.BUILD_INFO_COLLECTION,
            SkyFunctions.ACTION_EXECUTION)));
  }

  /**
   * Removes ConfigurationFragmentValuess and ConfigurationCollectionValues from the cache.
   */
  @VisibleForTesting
  public void invalidateConfigurationCollection() {
    invalidate(SkyFunctionName.functionIsIn(ImmutableSet.of(SkyFunctions.CONFIGURATION_FRAGMENT,
            SkyFunctions.CONFIGURATION_COLLECTION)));
  }

  /**
   * Deletes all ConfiguredTarget values from the Skyframe cache.
   *
   * <p>After the execution of this method all invalidated and marked for deletion values
   * (and the values depending on them) will be deleted from the cache.
   *
   * <p>WARNING: Note that a call to this method leaves legacy data inconsistent with Skyframe.
   * The next build should clear the legacy caches.
   */
  private void dropConfiguredTargetsNow() {
    dropConfiguredTargets();
    // Run the invalidator to actually delete the values.
    try {
      progressReceiver.ignoreInvalidations = true;
      callUninterruptibly(new Callable<Void>() {
        @Override
        public Void call() throws InterruptedException {
          buildDriver.evaluate(ImmutableList.<SkyKey>of(), false,
              ResourceUsage.getAvailableProcessors(), reporter);
          return null;
        }
      });
    } catch (Exception e) {
      throw new IllegalStateException(e);
    } finally {
      progressReceiver.ignoreInvalidations = false;
    }
  }

  /**
   * Save memory by removing references to configured targets and actions in Skyframe.
   *
   * <p>These values must be recreated on subsequent builds. We do not clear the top-level target
   * values, since their configured targets are needed for the target completion middleman values.
   *
   * <p>The values are not deleted during this method call, because they are needed for the
   * execution phase. Instead, their data is cleared. The next build will delete the values (and
   * recreate them if necessary).
   */
  private void discardAnalysisCache(Collection<ConfiguredTarget> topLevelTargets) {
    lastAnalysisDiscarded = true;
    for (Map.Entry<SkyKey, SkyValue> entry : memoizingEvaluator.getValues().entrySet()) {
      if (!entry.getKey().functionName().equals(SkyFunctions.CONFIGURED_TARGET)) {
        continue;
      }
      ConfiguredTargetValue ctValue = (ConfiguredTargetValue) entry.getValue();
      // ctValue may be null if target was not successfully analyzed.
      if (ctValue != null && !topLevelTargets.contains(ctValue.getConfiguredTarget())) {
        ctValue.clear();
      }
    }
  }

  /**
   * Decides if graph edges should be stored for this build. If not, re-creates the graph to not
   * store graph edges. Necessary conditions to not store graph edges are:
   * (1) batch (since incremental builds are not possible);
   * (2) skyframe build (since otherwise the memory savings are too slight to bother);
   * (3) keep-going (since otherwise bubbling errors up may require edges of done nodes);
   * (4) discard_analysis_cache (since otherwise user isn't concerned about saving memory this way).
   */
  public void decideKeepIncrementalState(boolean batch, Options viewOptions) {
    Preconditions.checkState(!active);
    if (viewOptions == null) {
      // Some blaze commands don't include the view options. Don't bother with them.
      return;
    }
    if (skyframeBuild && batch && viewOptions.keepGoing && viewOptions.discardAnalysisCache) {
      Preconditions.checkState(keepGraphEdges, "May only be called once if successful");
      keepGraphEdges = false;
      // Graph will be recreated on next sync.
    }
  }

  public boolean hasIncrementalState() {
    return keepGraphEdges;
  }

  /**
   * Saves memory by clearing analysis objects from Skyframe. If using legacy execution, actually
   * deletes the relevant values. If using Skyframe execution, clears their data without deleting
   * them (they will be deleted on the next build).
   */
  public void clearAnalysisCache(Collection<ConfiguredTarget> topLevelTargets) {
    if (!skyframeBuild()) {
      dropConfiguredTargetsNow();
    } else {
      discardAnalysisCache(topLevelTargets);
    }
  }

  /**
   * Injects the contents of the computed tools/defaults package.
   */
  @VisibleForTesting
  public void setupDefaultPackage(String defaultsPackageContents) {
    BuildVariableValue.DEFAULTS_PACKAGE_CONTENTS.set(recordingDiffer, defaultsPackageContents);
  }

  /**
   * Injects the top-level artifact options.
   */
  public void injectTopLevelContext(TopLevelArtifactContext options) {
    Preconditions.checkState(skyframeBuild(), "Only inject top-level context in Skyframe full");
    BuildVariableValue.TOP_LEVEL_CONTEXT.set(recordingDiffer, options);
  }

  public void injectWorkspaceStatusData() {
    BuildVariableValue.WORKSPACE_STATUS_KEY.set(recordingDiffer,
        workspaceStatusActionFactory.createWorkspaceStatusAction(
            artifactFactory.get(), WorkspaceStatusValue.ARTIFACT_OWNER, buildId));
  }

  /**
   * Sets the default visibility.
   */
  private void setDefaultVisibility(RuleVisibility defaultVisibility) {
    BuildVariableValue.DEFAULT_VISIBILITY.set(recordingDiffer, defaultVisibility);
  }

  /**
   * Injects the build info factory map that will be used when constructing build info
   * actions/artifacts. Unchanged across the life of the Blaze server, although it must be injected
   * each time the evaluator is created.
   */
  private void injectBuildInfoFactories() {
    ImmutableMap.Builder<BuildInfoKey, BuildInfoFactory> factoryMapBuilder =
        ImmutableMap.builder();
    for (BuildInfoFactory factory : buildInfoFactories) {
      factoryMapBuilder.put(factory.getKey(), factory);
    }
    BuildVariableValue.BUILD_INFO_FACTORIES.set(recordingDiffer, factoryMapBuilder.build());
  }

  private void setShowLoadingProgress(boolean showLoadingProgressValue) {
    showLoadingProgress.set(showLoadingProgressValue);
  }

  @VisibleForTesting
  public void setCommandId(UUID commandId) {
    BuildVariableValue.BUILD_ID.set(recordingDiffer, commandId);
    buildId.val = commandId;
  }

  private void invalidateDeletedPackages(Iterable<String> deletedPackages) {
    ArrayList<SkyKey> packagesToInvalidate = Lists.newArrayList();
    for (String deletedPackage : deletedPackages) {
      PathFragment pathFragment = new PathFragment(deletedPackage);
      packagesToInvalidate.add(PackageLookupValue.key(pathFragment));
    }
    recordingDiffer.invalidate(packagesToInvalidate);
  }

  /** Returns the build-info.txt and build-changelist.txt artifacts. */
  public Collection<Artifact> getWorkspaceStatusArtifacts() throws InterruptedException {
    // Should already be present, unless the user didn't request any targets for analysis.
    EvaluationResult<WorkspaceStatusValue> result = buildDriver.evaluate(
        ImmutableList.of(WorkspaceStatusValue.SKY_KEY), /*keepGoing=*/false, /*numThreads=*/1,
        reporter);
    WorkspaceStatusValue value = result.get(WorkspaceStatusValue.SKY_KEY);
    return ImmutableList.of(value.getStableArtifact(), value.getVolatileArtifact());
  }

  @VisibleForTesting
  public WorkspaceStatusAction getLastWorkspaceStatusActionForTesting() {
    BuildVariableValue value = (BuildVariableValue) buildDriver.getGraphForTesting()
        .getExistingValueForTesting(BuildVariableValue.WORKSPACE_STATUS_KEY.getKeyForTesting());
    return (WorkspaceStatusAction) value.get();
  }

  /**
   * Informs user about number of modified files (source and output files).
   */
  // Note, that number of modified files in some cases can be bigger than actual number of
  // modified files for targets in current request. Skyframe checks for modification all files
  // from previous requests.
  public void informAboutNumberOfModifiedFiles() {
    LOG.info(String.format("Found %d modified files from last build", modifiedFiles));
  }

  public Reporter getReporter() {
    return reporter;
  }

  public EventBus getEventBus() {
    return eventBus.get();
  }

  @VisibleForTesting
  ImmutableList<Path> getPathEntries() {
    return pkgLocator.get().getPathEntries();
  }

  private void invalidate(Predicate<SkyKey> pred) {
    recordingDiffer.invalidate(Iterables.filter(memoizingEvaluator.getValues().keySet(), pred));
  }

  /**
   * Partitions the given filesystem values based on which package path root they are under.
   * Returns a {@link Multimap} {@code m} such that {@code m.containsEntry(k, pe)} is true for
   * each filesystem valuekey {@code k} under a package path root {@code pe}. Note that values not
   * under a package path root are not present in the returned {@link Multimap}; these values are
   * unconditionally checked for changes on each incremental build.
   */
  private static Multimap<Path, SkyKey> partitionSkyKeysByPackagePathEntry(
      Set<Path> pkgRoots, Iterable<SkyKey> filesystemSkyKeys) {
    ImmutableSetMultimap.Builder<Path, SkyKey> multimapBuilder =
        ImmutableSetMultimap.builder();
    for (SkyKey key : filesystemSkyKeys) {
      Preconditions.checkState(key.functionName() == SkyFunctions.FILE_STATE
          || key.functionName() == SkyFunctions.DIRECTORY_LISTING_STATE, key);
      Path root = ((RootedPath) key.argument()).getRoot();
      if (pkgRoots.contains(root)) {
        multimapBuilder.put(root, key);
      }
      // We don't need to worry about FileStateValues for external files because they have a
      // dependency on the build_id and so they get invalidated each build.
    }
    return multimapBuilder.build();
  }

  private static Iterable<SkyKey> getSkyKeysPotentiallyAffected(
      Iterable<PathFragment> modifiedSourceFiles, final Path pathEntry) {
    // TODO(bazel-team): change ModifiedFileSet to work with RootedPaths instead of PathFragments.
    Iterable<SkyKey> fileStateSkyKeys = Iterables.transform(modifiedSourceFiles,
        new Function<PathFragment, SkyKey>() {
          @Override
          public SkyKey apply(PathFragment pathFragment) {
            return FileStateValue.key(RootedPath.toRootedPath(pathEntry, pathFragment));
          }
        });
    // TODO(bazel-team): Strictly speaking, we only need to invalidate directory values when a file
    // has been created or deleted, not when it has been modified. Unfortunately we
    // do not have that information here, although fancy filesystems could provide it with a
    // hypothetically modified DiffAwareness interface.
    // TODO(bazel-team): Even if we don't have that information, we could avoid invalidating
    // directories when the state of a file does not change by statting them and comparing
    // the new filetype (nonexistent/file/symlink/directory) with the old one.
    Iterable<SkyKey> dirListingStateSkyKeys = Iterables.transform(modifiedSourceFiles,
        new Function<PathFragment, SkyKey>() {
          @Override
          public SkyKey apply(PathFragment pathFragment) {
            return DirectoryListingStateValue.key(RootedPath.toRootedPath(pathEntry,
                pathFragment.getParentDirectory()));
          }
        });
    return Iterables.concat(fileStateSkyKeys, dirListingStateSkyKeys);
  }

  private static int getNumberOfModifiedFiles(Iterable<SkyKey> modifiedValues) {
    // We are searching only for changed files, DirectoryListingValues don't depend on
    // child values, that's why they are invalidated separately
    return Iterables.size(Iterables.filter(modifiedValues,
        SkyFunctionName.functionIs(SkyFunctions.FILE_STATE)));
  }

  /**
   * Uses diff awareness on all the package paths to invalidate changed files.
   */
  @VisibleForTesting
  public void handleDiffs() throws InterruptedException {
    if (lastAnalysisDiscarded) {
      // Values were cleared last build, but they couldn't be deleted because they were needed for
      // the execution phase. We can delete them now.
      dropConfiguredTargetsNow();
      lastAnalysisDiscarded = false;
    }

    modifiedFiles = 0;
    Map<Path, ProcessableModifiedFileSet> modifiedFilesByPathEntry =
        Maps.newHashMap();
    Set<Pair<Path, ProcessableModifiedFileSet>> pathEntriesWithoutDiffInformation =
        Sets.newHashSet();
    for (Path pathEntry : pkgLocator.get().getPathEntries()) {
      ProcessableModifiedFileSet modifiedFileSet = diffAwarenessManager.getDiff(pathEntry);
      if (modifiedFileSet.getModifiedFileSet().treatEverythingAsModified()) {
        pathEntriesWithoutDiffInformation.add(Pair.of(pathEntry, modifiedFileSet));
      } else {
        modifiedFilesByPathEntry.put(pathEntry, modifiedFileSet);
      }
    }
    handleDiffsWithCompleteDiffInformation(modifiedFilesByPathEntry);
    handleDiffsWithMissingDiffInformation(pathEntriesWithoutDiffInformation);
  }

  /**
   * Invalidates files under path entries whose corresponding {@link DiffAwareness} gave an exact
   * diff. Removes entries from the given map as they are processed. All of the files need to be
   * invalidated, so the map should be empty upon completion of this function.
   */
  private void handleDiffsWithCompleteDiffInformation(
      Map<Path, ProcessableModifiedFileSet> modifiedFilesByPathEntry) {
    // It's important that the below code be uninterruptible, since we already promised to
    // invalidate these files.
    for (Path pathEntry : ImmutableSet.copyOf(modifiedFilesByPathEntry.keySet())) {
      ProcessableModifiedFileSet processableModifiedFileSet =
          modifiedFilesByPathEntry.get(pathEntry);
      ModifiedFileSet modifiedFileSet = processableModifiedFileSet.getModifiedFileSet();
      Preconditions.checkState(!modifiedFileSet.treatEverythingAsModified(), pathEntry);
      Iterable<SkyKey> dirtyValues = getSkyKeysPotentiallyAffected(
          modifiedFileSet.modifiedSourceFiles(), pathEntry);
      handleChangedFiles(new ImmutableDiff(dirtyValues, ImmutableMap.<SkyKey, SkyValue>of()));
      processableModifiedFileSet.markProcessed();
    }
  }

  /**
   * Finds and invalidates changed files under path entries whose corresponding
   * {@link DiffAwareness} said all files may have been modified.
   */
  private void handleDiffsWithMissingDiffInformation(
      Set<Pair<Path, ProcessableModifiedFileSet>> pathEntriesWithoutDiffInformation)
      throws InterruptedException {
    if (pathEntriesWithoutDiffInformation.isEmpty()) {
      return;
    }
    // Before running the FilesystemValueChecker, ensure that all values marked for invalidation
    // have actually been invalidated (recall that invalidation happens at the beginning of the
    // next evaluate() call), because checking those is a waste of time.
    buildDriver.evaluate(ImmutableList.<SkyKey>of(), false,
        DEFAULT_THREAD_COUNT, reporter);
    FilesystemValueChecker fsnc = new FilesystemValueChecker(memoizingEvaluator, tsgm);
    // We need to manually check for changes to known files. This entails finding all dirty file
    // system values under package roots for which we don't have diff information. If at least
    // one path entry doesn't have diff information, then we're going to have to iterate over
    // the skyframe values at least once no matter what so we might as well do so now and avoid
    // doing so more than once.
    Iterable<SkyKey> filesystemSkyKeys = fsnc.getFilesystemSkyKeys();
    // Partition by package path entry.
    Multimap<Path, SkyKey> skyKeysByPathEntry = partitionSkyKeysByPackagePathEntry(
        ImmutableSet.copyOf(pkgLocator.get().getPathEntries()), filesystemSkyKeys);
    // Contains all file system values that we need to check for dirtiness.
    List<Iterable<SkyKey>> valuesToCheckManually = Lists.newArrayList();
    for (Pair<Path, ProcessableModifiedFileSet> pair : pathEntriesWithoutDiffInformation) {
      Path pathEntry = pair.getFirst();
      valuesToCheckManually.add(skyKeysByPathEntry.get(pathEntry));
    }
    Diff diff = fsnc.getDirtyFilesystemValues(Iterables.concat(valuesToCheckManually));
    handleChangedFiles(diff);
    for (Pair<Path, ProcessableModifiedFileSet> pair : pathEntriesWithoutDiffInformation) {
      ProcessableModifiedFileSet processableModifiedFileSet = pair.getSecond();
      processableModifiedFileSet.markProcessed();
    }
  }

  private void handleChangedFiles(Diff diff) {
    recordingDiffer.invalidate(diff.changedKeysWithoutNewValues());
    recordingDiffer.inject(diff.changedKeysWithNewValues());
    modifiedFiles += getNumberOfModifiedFiles(diff.changedKeysWithoutNewValues());
    modifiedFiles += getNumberOfModifiedFiles(diff.changedKeysWithNewValues().keySet());
    if (skyframeBuild()) {
      incrementalBuildMonitor.accrue(diff.changedKeysWithoutNewValues());
      incrementalBuildMonitor.accrue(diff.changedKeysWithNewValues().keySet());
    }
  }

  /**
   * Sets the packages that should be treated as deleted and ignored.
   */
  @VisibleForTesting  // productionVisibility = Visibility.PRIVATE
  public void setDeletedPackages(Iterable<String> pkgs) {
    // Invalidate the old deletedPackages as they may exist now.
    invalidateDeletedPackages(deletedPackages.get());
    deletedPackages.set(ImmutableSet.copyOf(pkgs));
    // Invalidate the new deletedPackages as we need to pretend that they don't exist now.
    invalidateDeletedPackages(deletedPackages.get());
  }

  /**
   * Prepares the evaluator for loading.
   *
   * <p>MUST be run before every incremental build.
   */
  @VisibleForTesting  // productionVisibility = Visibility.PRIVATE
  public void preparePackageLoading(PathPackageLocator pkgLocator, RuleVisibility defaultVisibility,
      boolean showLoadingProgress,
      String defaultsPackageContents, UUID commandId) {
    Preconditions.checkNotNull(pkgLocator);
    setActive(true);

    setCommandId(commandId);
    setShowLoadingProgress(showLoadingProgress);
    setDefaultVisibility(defaultVisibility);
    setupDefaultPackage(defaultsPackageContents);
    setPackageLocator(pkgLocator);

    syscalls.set(new PerBuildSyscallCache());
    checkPreprocessorFactory();
    emittedEventState.clear();

    // If the PackageFunction was interrupted, there may be stale entries here.
    packageFunctionCache.clear();
    numPackagesLoaded.set(0);

    // Reset the stateful SkyframeCycleReporter, which contains cycles from last run.
    cyclesReporter.set(createCyclesReporter());
  }

  /**
   * The value types whose builders have direct access to the package locator. They need to be
   * invalidated if the package locator changes.
   */
  private static final Set<SkyFunctionName> PACKAGE_LOCATOR_DEPENDENT_VALUES = ImmutableSet.of(
          SkyFunctions.FILE_STATE,
          SkyFunctions.FILE,
          SkyFunctions.DIRECTORY_LISTING_STATE,
          SkyFunctions.PACKAGE_LOOKUP,
          SkyFunctions.TARGET_PATTERN);

  @SuppressWarnings("unchecked")
  private void setPackageLocator(PathPackageLocator pkgLocator) {
    PathPackageLocator oldLocator = this.pkgLocator.getAndSet(pkgLocator);
    if ((oldLocator == null || !oldLocator.getPathEntries().equals(pkgLocator.getPathEntries()))) {
      invalidate(SkyFunctionName.functionIsIn(PACKAGE_LOCATOR_DEPENDENT_VALUES));

      // The package path is read not only by SkyFunctions but also by some other code paths.
      // We need to take additional steps to keep the corresponding data structures in sync.
      // (Some of the additional steps are carried out by ConfiguredTargetValueInvalidationListener,
      // and some by BuildView#buildHasIncompatiblePackageRoots and #updateSkyframe.)

      diffAwarenessManager.setPathEntries(pkgLocator.getPathEntries());
    }
  }

  private void checkPreprocessorFactory() {
    if (!preprocessorFactory.isStillValid()) {
      Preprocessor.Factory newPreprocessorFactory = preprocessorFactorySupplier.getFactory(
          packageManager);
      invalidate(SkyFunctionName.functionIs(SkyFunctions.PACKAGE));
      pkgFactory.setPreprocessorFactory(newPreprocessorFactory);
      preprocessorFactory = newPreprocessorFactory;
    }
  }

  /**
   * Specifies the current {@link SkyframeBuildView} instance. This should only be set once over the
   * lifetime of the Blaze server, except in tests.
   */
  public void setSkyframeBuildView(SkyframeBuildView skyframeBuildView) {
    this.skyframeBuildView = skyframeBuildView;
    setConfigurationSkyKey(configurationSkyKey);
    this.artifactFactory.val = skyframeBuildView.getArtifactFactory();
    if (skyframeBuildView.getWarningListener() != null) {
      setErrorEventListener(skyframeBuildView.getWarningListener());
    }
  }

  /**
   * Sets the eventBus to use for posting events.
   */
  public void setEventBus(EventBus eventBus) {
    this.eventBus.set(eventBus);
  }

  /**
   * Sets the eventHandler to use for reporting errors.
   */
  public void setErrorEventListener(EventHandler eventHandler) {
    this.errorEventListener = eventHandler;
  }

  /**
   * Sets the path for action log buffers.
   */
  public void setActionOutputRoot(Path actionOutputRoot) {
    Preconditions.checkNotNull(actionOutputRoot);
    this.actionLogBufferPathGenerator = new ActionLogBufferPathGenerator(actionOutputRoot);
    this.skyframeActionExecutor.setActionLogBufferPathGenerator(actionLogBufferPathGenerator);
  }

  private void setConfigurationSkyKey(SkyKey skyKey) {
    this.configurationSkyKey = skyKey;
    if (skyframeBuildView != null) {
      skyframeBuildView.setConfigurationSkyKey(skyKey);
    }
  }

  @VisibleForTesting
  public void setConfigurationDataForTesting(BuildOptions options,
      BlazeDirectories directories, ConfigurationFactory configurationFactory) {
    SkyKey skyKey = ConfigurationCollectionValue.key(options, ImmutableSet.<String>of());
    setConfigurationSkyKey(skyKey);
    BuildVariableValue.BLAZE_DIRECTORIES.set(recordingDiffer, directories);
    this.configurationFactory.val = configurationFactory;
    this.configurationFragments.val = ImmutableList.copyOf(configurationFactory.getFactories());
  }

  /**
   * Asks the Skyframe evaluator to build the value for BuildConfigurationCollection and
   * returns result. Also invalidates {@link BuildVariableValue#TEST_ENVIRONMENT_VARIABLES} and
   * {@link BuildVariableValue#BLAZE_DIRECTORIES} if they have changed.
   */
  public BuildConfigurationCollection createConfigurations(boolean keepGoing,
      ConfigurationFactory configurationFactory, BuildConfigurationKey configurationKey)
      throws InvalidConfigurationException, InterruptedException {

    this.clientEnv.val = configurationKey.getClientEnv();
    this.configurationFactory.val = configurationFactory;
    this.configurationFragments.val = ImmutableList.copyOf(configurationFactory.getFactories());
    BuildOptions buildOptions = configurationKey.getBuildOptions();
    Map<String, String> testEnv = BuildConfiguration.getTestEnv(
        buildOptions.get(BuildConfiguration.Options.class).testEnvironment,
        configurationKey.getClientEnv());
    // TODO(bazel-team): find a way to use only BuildConfigurationKey instead of
    // TestEnvironmentVariables and BlazeDirectories. There is a problem only with
    // TestEnvironmentVariables because BuildConfigurationKey stores client environment variables
    // and we don't want to rebuild everything when any variable changes.
    BuildVariableValue.TEST_ENVIRONMENT_VARIABLES.set(recordingDiffer, testEnv);
    BuildVariableValue.BLAZE_DIRECTORIES.set(recordingDiffer, configurationKey.getDirectories());

    SkyKey skyKey = ConfigurationCollectionValue.key(configurationKey.getBuildOptions(),
        configurationKey.getMultiCpu());
    setConfigurationSkyKey(skyKey);
    EvaluationResult<ConfigurationCollectionValue> result = buildDriver.evaluate(
            Arrays.asList(skyKey), keepGoing, DEFAULT_THREAD_COUNT, errorEventListener);
    if (result.hasError()) {
      Throwable e = result.getError(skyKey).getException();
      Throwables.propagateIfInstanceOf(e, InvalidConfigurationException.class);
      throw new IllegalStateException(
          "Unknown error during ConfigurationCollectionValue evaluation", e);
    }
    Preconditions.checkState(result.values().size() == 1,
        "Result of evaluate() must contain exactly one value " + result);
    return Iterables.getOnlyElement(result.values()).getConfigurationCollection();
  }

  private Iterable<ActionLookupValue> getActionLookupValues() {
    // This filter keeps subclasses of ActionLookupValue.
    return Iterables.filter(memoizingEvaluator.getDoneValues().values(), ActionLookupValue.class);
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  Map<SkyKey, ActionLookupValue> getActionLookupValueMap() {
    return (Map) Maps.filterValues(memoizingEvaluator.getDoneValues(),
        Predicates.instanceOf(ActionLookupValue.class));
  }

  /**
   * Checks the actions in Skyframe for conflicts between their output artifacts. Delegates to
   * {@link SkyframeActionExecutor#findAndStoreArtifactConflicts} to do the work, since any
   * conflicts found will only be reported during execution.
   */
  ImmutableMap<Action, Exception> findArtifactConflicts() throws InterruptedException {
    Preconditions.checkState(skyframeBuild);
    if (skyframeBuildView.isSomeConfiguredTargetEvaluated()
        || skyframeBuildView.isSomeConfiguredTargetInvalidated()) {
      // This operation is somewhat expensive, so we only do it if the graph might have changed in
      // some way -- either we analyzed a new target or we invalidated an old one.
      skyframeActionExecutor.findAndStoreArtifactConflicts(getActionLookupValues());
      skyframeBuildView.resetEvaluatedConfiguredTargetFlag();
      // The invalidated configured targets flag will be reset later in the evaluate() call.
    }
    return skyframeActionExecutor.badActions();
  }

  /**
   * Asks the Skyframe evaluator to build the values corresponding to the given artifacts.
   *
   * <p>The returned artifacts should be built and present on the filesystem after the call
   * completes.
   */
  public EvaluationResult<ArtifactValue> buildArtifacts(
      Executor executor,
      Set<Artifact> artifacts,
      boolean keepGoing,
      boolean explain,
      int numJobs,
      ActionCacheChecker actionCacheChecker,
      @Nullable EvaluationProgressReceiver executionProgressReceiver) throws InterruptedException {
    checkActive();
    Preconditions.checkState(actionLogBufferPathGenerator != null);

    skyframeActionExecutor.prepareForExecution(executor, keepGoing, explain, actionCacheChecker);

    resourceManager.resetResourceUsage();
    try {
      progressReceiver.executionProgressReceiver = executionProgressReceiver;
      return buildDriver.evaluate(ArtifactValue.mandatoryKeys(artifacts), keepGoing,
          numJobs, errorEventListener);
    } finally {
      progressReceiver.executionProgressReceiver = null;
      // Also releases thread locks.
      resourceManager.resetResourceUsage();
      skyframeActionExecutor.executionOver();
    }
  }

  @VisibleForTesting
  public void prepareBuildingForTestingOnly(Executor executor, boolean keepGoing, boolean explain,
                                            ActionCacheChecker checker) {
    skyframeActionExecutor.prepareForExecution(executor, keepGoing, explain, checker);
  }

  EvaluationResult<TargetPatternValue> targetPatterns(Iterable<SkyKey> patternSkyKeys,
      boolean keepGoing, EventHandler eventHandler) throws InterruptedException {
    checkActive();
    return buildDriver.evaluate(patternSkyKeys, keepGoing, DEFAULT_THREAD_COUNT,
        eventHandler);
  }

  /**
   * Returns the {@link ConfiguredTarget}s corresponding to the given keys.
   *
   * <p>For use for legacy support from {@code BuildView} only.
   */
  @ThreadSafety.ThreadSafe
  public ImmutableList<ConfiguredTarget> getConfiguredTargets(
      Iterable<LabelAndConfiguration> lacs) {
    checkActive();
    if (skyframeBuildView == null) {
      // If build view has not yet been initialized, no configured targets can have been created.
      // This is most likely to happen after a failed loading phase.
      return ImmutableList.of();
    }
    final Collection<SkyKey> skyKeys = ConfiguredTargetValue.keys(lacs);
    EvaluationResult<SkyValue> result;
    try {
      result = callUninterruptibly(new Callable<EvaluationResult<SkyValue>>() {
        @Override
        public EvaluationResult<SkyValue> call() throws Exception {
          synchronized (valueLookupLock) {
            try {
              skyframeBuildView.enableAnalysis(true);
              return buildDriver.evaluate(skyKeys, false, DEFAULT_THREAD_COUNT,
                  errorEventListener);
            } finally {
              skyframeBuildView.enableAnalysis(false);
            }
          }
        }
      });
    } catch (Exception e) {
      throw new IllegalStateException(e);  // Should never happen.
    }

    ImmutableList.Builder<ConfiguredTarget> cts = ImmutableList.builder();
    for (SkyValue value : result.values()) {
      ConfiguredTargetValue ctValue = (ConfiguredTargetValue) value;
      cts.add(ctValue.getConfiguredTarget());
    }
    return cts.build();
  }

  /**
   * Returns a particular configured target.
   *
   * <p>Used only for testing.
   */
  @VisibleForTesting
  @Nullable
  public ConfiguredTarget getConfiguredTargetForTesting(
      Label label, BuildConfiguration configuration) {
    if (memoizingEvaluator.getExistingValueForTesting(
        BuildVariableValue.WORKSPACE_STATUS_KEY.getKeyForTesting()) == null) {
      injectWorkspaceStatusData();
    }
    return Iterables.getFirst(getConfiguredTargets(ImmutableList.of(
        new LabelAndConfiguration(label, configuration))), null);
  }

  /**
   * Invalidates Skyframe values corresponding to the given set of modified files under the given
   * path entry.
   *
   * <p>May throw an {@link InterruptedException}, which means that no values have been invalidated.
   */
  @VisibleForTesting
  public void invalidateFilesUnderPathForTesting(ModifiedFileSet modifiedFileSet, Path pathEntry)
      throws InterruptedException {
    if (lastAnalysisDiscarded) {
      // Values were cleared last build, but they couldn't be deleted because they were needed for
      // the execution phase. We can delete them now.
      dropConfiguredTargetsNow();
      lastAnalysisDiscarded = false;
    }
    Iterable<SkyKey> keys;
    if (modifiedFileSet.treatEverythingAsModified()) {
      Diff diff = new FilesystemValueChecker(memoizingEvaluator, tsgm).getDirtyFilesystemSkyKeys();
      keys = diff.changedKeysWithoutNewValues();
      recordingDiffer.inject(diff.changedKeysWithNewValues());
    } else {
      keys = getSkyKeysPotentiallyAffected(modifiedFileSet.modifiedSourceFiles(), pathEntry);
    }
    syscalls.set(new PerBuildSyscallCache());
    recordingDiffer.invalidate(keys);
    // Blaze invalidates (transient) errors on every build.
    invalidateErrors();
  }

  /**
   * Invalidates SkyFrame values that may have failed for transient reasons.
   */
  @VisibleForTesting  // productionVisibility = Visibility.PRIVATE
  public void invalidateErrors() {
    checkActive();
    recordingDiffer.invalidateErrors();
  }

  @VisibleForTesting
  public TimestampGranularityMonitor getTimestampGranularityMonitorForTesting() {
    return tsgm;
  }

  /**
   * Configures a given set of configured targets.
   */
  public EvaluationResult<ConfiguredTargetValue> configureTargets(
      List<LabelAndConfiguration> values, boolean keepGoing) throws InterruptedException {
    checkActive();

    // Make sure to not run too many analysis threads. This can cause memory thrashing.
    return buildDriver.evaluate(ConfiguredTargetValue.keys(values), keepGoing,
        ResourceUsage.getAvailableProcessors(), errorEventListener);
  }

  /**
   * Post-process the targets. Values in the EvaluationResult are known to be transitively
   * error-free from action conflicts.
   */
  public EvaluationResult<PostConfiguredTargetValue> postConfigureTargets(
      List<LabelAndConfiguration> values, boolean keepGoing,
      ImmutableMap<Action, Exception> badActions) throws InterruptedException {
    checkActive();
    BuildVariableValue.BAD_ACTIONS.set(recordingDiffer, badActions);
    // Make sure to not run too many analysis threads. This can cause memory thrashing.
    EvaluationResult<PostConfiguredTargetValue> result =
        buildDriver.evaluate(PostConfiguredTargetValue.keys(values), keepGoing,
            ResourceUsage.getAvailableProcessors(), errorEventListener);

    // Remove all post-configured target values immediately for memory efficiency. We are OK with
    // this mini-phase being non-incremental as the failure mode of action conflict is rare.
    memoizingEvaluator.delete(SkyFunctionName.functionIs(SkyFunctions.POST_CONFIGURED_TARGET));

    return result;
  }

  /**
   * Returns a Skyframe-based {@link SkyframeTransitivePackageLoader} implementation.
   */
  @VisibleForTesting
  public TransitivePackageLoader pkgLoader() {
    checkActive();
    return new SkyframeLabelVisitor(new SkyframeTransitivePackageLoader(), cyclesReporter);
  }

  class SkyframeTransitivePackageLoader {
    /**
     * Loads the specified {@link TransitiveTargetValue}s.
     */
    EvaluationResult<TransitiveTargetValue> loadTransitiveTargets(
        Iterable<Target> targetsToVisit, Iterable<Label> labelsToVisit, boolean keepGoing)
        throws InterruptedException {
      List<SkyKey> valueNames = new ArrayList<>();
      for (Target target : targetsToVisit) {
        valueNames.add(TransitiveTargetValue.key(target.getLabel()));
      }
      for (Label label : labelsToVisit) {
        valueNames.add(TransitiveTargetValue.key(label));
      }

      return buildDriver.evaluate(valueNames, keepGoing, DEFAULT_THREAD_COUNT,
          errorEventListener);
    }

    public Set<Package> retrievePackages(Set<PathFragment> packageNames) {
      final List<SkyKey> valueNames = new ArrayList<>();
      for (PathFragment pkgName : packageNames) {
        valueNames.add(PackageValue.key(pkgName));
      }

      try {
        return callUninterruptibly(new Callable<Set<Package>>() {
          @Override
          public Set<Package> call() throws Exception {
            EvaluationResult<PackageValue> result = buildDriver.evaluate(
                valueNames, false, ResourceUsage.getAvailableProcessors(), errorEventListener);
            Preconditions.checkState(!result.hasError(),
                "unexpected errors: %s", result.errorMap());
            Set<Package> packages = Sets.newHashSet();
            for (PackageValue value : result.values()) {
              packages.add(value.getPackage());
            }
            return packages;
          }
        });
      } catch (Exception e) {
        throw new IllegalStateException(e);
      }

    }
  }

  /**
   * Returns the generating {@link Action} of the given {@link Artifact}.
   *
   * <p>For use for legacy support from {@code BuildView} only.
   */
  @ThreadSafety.ThreadSafe
  public Action getGeneratingAction(final Artifact artifact) {
    if (artifact.isSourceArtifact()) {
      return null;
    }

    try {
      return callUninterruptibly(new Callable<Action>() {
        @Override
        public Action call() throws InterruptedException {
          ArtifactOwner artifactOwner = artifact.getArtifactOwner();
          Preconditions.checkState(artifactOwner instanceof ActionLookupValue.ActionLookupKey,
              "%s %s", artifact, artifactOwner);
          SkyKey actionLookupKey =
              ActionLookupValue.key((ActionLookupValue.ActionLookupKey) artifactOwner);

          synchronized (valueLookupLock) {
            // Note that this will crash (attempting to run a configured target value builder after
            // analysis) after a failed --nokeep_going analysis in which the configured target that
            // failed was a (transitive) dependency of the configured target that should generate
            // this action. We don't expect callers to query generating actions in such cases.
            EvaluationResult<ActionLookupValue> result = buildDriver.evaluate(
                ImmutableList.of(actionLookupKey), false, ResourceUsage.getAvailableProcessors(),
                errorEventListener);
            return result.hasError()
                ? null
                : result.get(actionLookupKey).getGeneratingAction(artifact);
          }
        }
      });
    } catch (Exception e) {
      throw new IllegalStateException("Error getting generating action: " + artifact.prettyPrint(),
          e);
    }
  }

  public PackageManager getPackageManager() {
    return packageManager;
  }

  class SkyframePackageLoader {
    /**
     * Looks up a particular package (used after the loading phase).
     *
     * <p>Note that this method needs to be synchronized since InMemoryMemoizingEvaluator.evaluate()
     * method does not support concurrent calls.
     */
    Package getPackage(EventHandler eventHandler, String pkgName) throws InterruptedException,
        NoSuchPackageException {
      synchronized (valueLookupLock) {
        SkyKey key = PackageValue.key(new PathFragment(pkgName));
        EvaluationResult<PackageValue> result =
            buildDriver.evaluate(ImmutableList.of(key), false,
                DEFAULT_THREAD_COUNT, eventHandler);
        if (result.hasError()) {
          ErrorInfo error = result.getError();
          if (!Iterables.isEmpty(error.getCycleInfo())) {
            reportCycles(result.getError().getCycleInfo(), key);
            // This can only happen if a package is freshly loaded outside of the target parsing
            // or loading phase
            throw new BuildFileContainsErrorsException(pkgName,
                "Cycle encountered while loading package " + pkgName);
          }
          Throwable e = error.getException();
          // PackageFunction should be catching, swallowing, and rethrowing all transitive
          // errors as NoSuchPackageExceptions.
          Throwables.propagateIfInstanceOf(e, NoSuchPackageException.class);
          throw new IllegalStateException("Unexpected Exception type from PackageValue for '"
              + pkgName + "'' with root causes: " + Iterables.toString(error.getRootCauses()), e);
        }
        return result.get(key).getPackage();
      }
    }

    Package getLoadedPackage(final String pkgName) throws NoSuchPackageException {
      // Note that in Skyframe there is no way to tell if the package has been loaded before or not,
      // so this will never throw for packages that are not loaded. However, no code currently
      // relies on having the exception thrown.
      try {
        return callUninterruptibly(new Callable<Package>() {
          @Override
          public Package call() throws Exception {
            return getPackage(errorEventListener, pkgName);
          }
        });
      } catch (NoSuchPackageException e) {
        if (e.getPackage() != null) {
          return e.getPackage();
        }
        throw e;
      } catch (Exception e) {
        throw new IllegalStateException(e);  // Should never happen.
      }
    }

    /**
     * Returns whether the given package should be consider deleted and thus should be ignored.
     */
    public boolean isPackageDeleted(String packageName) {
      return deletedPackages.get().contains(packageName);
    }

    /** Same as {@link PackageManager#partiallyClear}. */
    void partiallyClear() {
      packageFunctionCache.clear();
    }
  }

  /**
   * Calls the given callable uninterruptibly.
   *
   * <p>If the callable throws {@link InterruptedException}, calls it again, until the callable
   * returns a result. Sets the {@code currentThread().interrupted()} bit if the callable threw
   * {@link InterruptedException} at least once.
   *
   * <p>This is almost identical to {@code Uninterruptibles#getUninterruptibly}.
   */
  private static final <T> T callUninterruptibly(Callable<T> callable) throws Exception {
    boolean interrupted = false;
    try {
      while (true) {
        try {
          return callable.call();
        } catch (InterruptedException e) {
          interrupted = true;
        }
      }
    } finally {
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
    }
  }

  @VisibleForTesting
  public MemoizingEvaluator getEvaluatorForTesting() {
    return memoizingEvaluator;
  }

  @VisibleForTesting
  public RecordingDifferencer getDifferencerForTesting() {
    return recordingDiffer;
  }

  /**
   * Returns true if the old set of Packages is a subset or superset of the new one.
   *
   * <p>Compares the names of packages instead of the Package objects themselves (Package doesn't
   * yet override #equals). Since packages store their names as a String rather than a Label, it's
   * easier to use strings here.
   */
  @VisibleForTesting
  static boolean isBuildSubsetOrSupersetOfPreviousBuild(Set<PathFragment> oldPackages,
      Set<PathFragment> newPackages) {
    if (newPackages.size() <= oldPackages.size()) {
      return Sets.difference(newPackages, oldPackages).isEmpty();
    } else if (oldPackages.size() < newPackages.size()) {
      // No need to check for <= here, since the first branch does that already.
      // If size(A) = size(B), then then A\B = 0 iff B\A = 0
      return Sets.difference(oldPackages, newPackages).isEmpty();
    } else {
      return false;
    }
  }

  /**
   * Stores the set of loaded packages and, if needed, evicts ConfiguredTarget values.
   *
   * <p>The set represents all packages from the transitive closure of the top-level targets from
   * the latest build.
   */
  @ThreadCompatible
  public void updateLoadedPackageSet(Set<PathFragment> loadedPackages) {
    Preconditions.checkState(valueCacheEvictionLimit >= 0,
        "should have called setMinLoadedPkgCountForCtValueEviction earlier");

    // Make a copy to avoid nesting SetView objects. It also computes size(), which we need below.
    Set<PathFragment> union = ImmutableSet.copyOf(Sets.union(allLoadedPackages, loadedPackages));

    if (union.size() < valueCacheEvictionLimit ||
        isBuildSubsetOrSupersetOfPreviousBuild(allLoadedPackages, loadedPackages)) {
      allLoadedPackages = union;
    } else {
      dropConfiguredTargets();
      allLoadedPackages = loadedPackages;
    }
  }

  public void sync(PackageCacheOptions packageCacheOptions, Path workingDirectory,
      String defaultsPackageContents, UUID commandId) throws InterruptedException {
    PathPackageLocator packageLocator = PathPackageLocator.create(
        packageCacheOptions.packagePath, getReporter(), directories.getWorkspace(),
        workingDirectory);

    preparePackageLoading(packageLocator,
        packageCacheOptions.defaultVisibility, packageCacheOptions.showLoadingProgress,
        defaultsPackageContents, commandId);
    setDeletedPackages(ImmutableSet.copyOf(packageCacheOptions.deletedPackages));
    this.valueCacheEvictionLimit = packageCacheOptions.minLoadedPkgCountForCtNodeEviction;

    incrementalBuildMonitor = new SkyframeIncrementalBuildMonitor();
    invalidateErrors();
    handleDiffs();
  }

  private CyclesReporter createCyclesReporter() {
    return new CyclesReporter(new TransitiveTargetCycleReporter(packageManager),
        new ConfiguredTargetCycleReporter(packageManager));
  }

  CyclesReporter getCyclesReporter() {
    return cyclesReporter.get();
  }

  /** Convenience method with same semantics as {@link CyclesReporter#reportCycles}. */
  public void reportCycles(Iterable<CycleInfo> cycles, SkyKey topLevelKey) {
    getCyclesReporter().reportCycles(cycles, topLevelKey, errorEventListener);
  }

  public void setActionExecutionProgressReportingObjects(@Nullable ProgressSupplier supplier,
      @Nullable ActionCompletedReceiver completionReceiver,
      @Nullable ActionExecutionStatusReporter statusReporter) {
    skyframeActionExecutor.setActionExecutionProgressReportingObjects(supplier, completionReceiver);
    this.statusReporterRef.set(statusReporter);
  }

  /**
   * This should be called at most once in the lifetime of the SkyframeExecutor (except for tests),
   * and it should be called before the execution phase.
   */
  void setArtifactFactoryAndBinTools(ArtifactFactory artifactFactory, BinTools binTools) {
    this.artifactFactory.val = artifactFactory;
    this.binTools = binTools;
  }

  public void prepareExecution(boolean checkOutputFiles) throws AbruptExitException,
      InterruptedException {
    Preconditions.checkState(skyframeBuild(),
        "Cannot prepare execution phase if not using Skyframe full");
    maybeInjectEmbeddedArtifacts();

    if (checkOutputFiles) {
      // Detect external modifications in the output tree.
      FilesystemValueChecker fsnc = new FilesystemValueChecker(memoizingEvaluator, tsgm);
      recordingDiffer.invalidate(fsnc.getDirtyActionValues(batchStatter));
      modifiedFiles += fsnc.getNumberOfModifiedOutputFiles();
    }
  }

  @VisibleForTesting void maybeInjectEmbeddedArtifacts() throws AbruptExitException {
    // The blaze client already ensures that the contents of the embedded binaries never change,
    // so we just need to make sure that the appropriate artifacts are present in the skyframe
    // graph.

    if (!needToInjectEmbeddedArtifacts) {
      return;
    }

    Preconditions.checkNotNull(artifactFactory.get());
    Preconditions.checkNotNull(binTools);
    Map<SkyKey, SkyValue> values = Maps.newHashMap();
    // Blaze separately handles the symlinks that target these binaries. See BinTools#setupTool.
    for (Artifact artifact : binTools.getAllEmbeddedArtifacts(artifactFactory.get())) {
      FileArtifactValue fileArtifactValue = null;
      try {
        fileArtifactValue = FileArtifactValue.create(artifact);
      } catch (IOException e) {
        // See ExtractData in blaze.cc.
        String message = "Error: corrupt installation: file " + artifact.getPath() + " missing. "
            + "Please remove '" + directories.getInstallBase() + "' and try again.";
        throw new AbruptExitException(message, ExitCode.LOCAL_ENVIRONMENTAL_ERROR, e);
      }
      values.put(ArtifactValue.key(artifact, /*isMandatory=*/true), fileArtifactValue);
    }
    recordingDiffer.inject(values);
    needToInjectEmbeddedArtifacts = false;
  }

  /**
   * Mark dirty values for deletion if they've been dirty for longer than N versions.
   *
   * <p>Specifying a value N means, if the current version is V and a value was dirtied (and
   * has remained so) in version U, and U + N &lt;= V, then the value will be marked for deletion
   * and purged in version V+1.
   */
  public void deleteOldNodes(long versionWindowForDirtyGc) {
    // TODO(bazel-team): perhaps we should come up with a separate GC class dedicated to maintaining
    // value garbage. If we ever do so, this logic should be moved there.
    memoizingEvaluator.deleteDirty(versionWindowForDirtyGc);
  }

  private class SkyframeProgressReceiver implements EvaluationProgressReceiver {

    /**
     * This flag is needed in order to avoid invalidating legacy data when we clear the
     * analysis cache because of --discard_analysis_cache flag. For that case we want to keep
     * the legacy data but get rid of the Skyframe data.
     */
    private boolean ignoreInvalidations = false;
    /** This receiver is only needed for execution, so it is null otherwise. */
    @Nullable EvaluationProgressReceiver executionProgressReceiver = null;

    @Override
    public void invalidated(SkyValue value, InvalidationState state) {
      if (ignoreInvalidations) {
        return;
      }
      if (skyframeBuildView != null) {
        skyframeBuildView.getInvalidationReceiver().invalidated(value, state);
      }
    }

    @Override
    public void enqueueing(SkyKey skyKey) {
      if (ignoreInvalidations) {
        return;
      }
      if (skyframeBuildView != null) {
        skyframeBuildView.getInvalidationReceiver().enqueueing(skyKey);
      }
      if (executionProgressReceiver != null) {
        executionProgressReceiver.enqueueing(skyKey);
      }
    }

    @Override
    public void evaluated(SkyKey skyKey, SkyValue value, EvaluationState state) {
      if (ignoreInvalidations) {
        return;
      }
      if (skyframeBuildView != null) {
        skyframeBuildView.getInvalidationReceiver().evaluated(skyKey, value, state);
      }
      if (executionProgressReceiver != null) {
        executionProgressReceiver.evaluated(skyKey, value, state);
      }
    }
  }

  /**
   * Supplier whose value can be changed by its "owner" (outer class). Unlike an {@link
   * AtomicReference}, clients cannot change its value.
   *
   * <p>This class must remain an inner class to allow only its outer class to modify its value.
   */
  private static class MutableSupplier<T> implements Supplier<T> {
    private T val;

    @Override
    public T get() {
      return val;
    }

    @SuppressWarnings("deprecation")  // MoreObjects.toStringHelper() is not in Guava
    @Override
    public String toString() {
      return Objects.toStringHelper(getClass())
          .add("val", val).toString();
    }
  }
}
