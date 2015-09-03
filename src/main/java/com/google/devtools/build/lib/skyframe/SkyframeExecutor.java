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
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.base.Throwables;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionExecutionContextFactory;
import com.google.devtools.build.lib.actions.ActionExecutionStatusReporter;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionLogBufferPathGenerator;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.PackageRootResolutionException;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.Aspect;
import com.google.devtools.build.lib.analysis.AspectWithParameters;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BuildView.Options;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.DependencyResolver.Dependency;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.Factory;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory.BuildInfoKey;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFactory;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.TransitivePackageLoader;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.skyframe.AspectValue.AspectKey;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.FileDirtinessChecker;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ActionCompletedReceiver;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ProgressSupplier;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.ResourceUsage;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.BuildDriver;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.Differencer;
import com.google.devtools.build.skyframe.Differencer.DiffWithDelta.Delta;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.ImmutableDiff;
import com.google.devtools.build.skyframe.Injectable;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator.EvaluatorSupplier;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph.WalkableGraphFactory;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.Callable;
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
public abstract class SkyframeExecutor implements WalkableGraphFactory {
  private final EvaluatorSupplier evaluatorSupplier;
  protected MemoizingEvaluator memoizingEvaluator;
  private final MemoizingEvaluator.EmittedEventState emittedEventState =
      new MemoizingEvaluator.EmittedEventState();
  protected final Reporter reporter;
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
  public static final int DEFAULT_THREAD_COUNT =
      // Reduce thread count while running tests of Bazel. Test cases are typically small, and large
      // thread pools vying for a relatively small number of CPU cores may induce non-optimal
      // performance.
      System.getenv("TEST_TMPDIR") == null ? 200 : 5;

  // Cache of partially constructed Package instances, stored between reruns of the PackageFunction
  // (because of missing dependencies, within the same evaluate() run) to avoid loading the same
  // package twice (first time loading to find subincludes and declare value dependencies).
  // TODO(bazel-team): remove this cache once we have skyframe-native package loading
  // [skyframe-loading]
  private final Cache<PackageIdentifier, Package.LegacyBuilder> packageFunctionCache =
      newPkgFunctionCache();

  private final AtomicInteger numPackagesLoaded = new AtomicInteger(0);

  protected SkyframeBuildView skyframeBuildView;
  private EventHandler errorEventListener;
  private ActionLogBufferPathGenerator actionLogBufferPathGenerator;

  protected BuildDriver buildDriver;

  // AtomicReferences are used here as mutable boxes shared with value builders.
  private final AtomicBoolean showLoadingProgress = new AtomicBoolean();
  protected final AtomicReference<UnixGlob.FilesystemCalls> syscalls =
      new AtomicReference<>(UnixGlob.DEFAULT_SYSCALLS);
  protected final AtomicReference<PathPackageLocator> pkgLocator =
      new AtomicReference<>();
  protected final AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages =
      new AtomicReference<>(ImmutableSet.<PackageIdentifier>of());
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

  protected boolean active = true;
  private final PackageManager packageManager;

  private final Preprocessor.Factory.Supplier preprocessorFactorySupplier;
  private Preprocessor.Factory preprocessorFactory;

  protected final TimestampGranularityMonitor tsgm;

  private final ResourceManager resourceManager;

  /** Used to lock evaluator on legacy calls to get existing values. */
  private final Object valueLookupLock = new Object();
  private final AtomicReference<ActionExecutionStatusReporter> statusReporterRef =
      new AtomicReference<>();
  private final SkyframeActionExecutor skyframeActionExecutor;
  private CompletionReceiver actionExecutionFunction;
  protected SkyframeProgressReceiver progressReceiver;
  private final AtomicReference<CyclesReporter> cyclesReporter = new AtomicReference<>();

  private final Set<Path> immutableDirectories;

  private BinTools binTools = null;
  private boolean needToInjectEmbeddedArtifacts = true;
  private boolean needToInjectPrecomputedValuesForAnalysis = true;
  protected int modifiedFiles;
  protected int outputDirtyFiles;
  protected int modifiedFilesDuringPreviousBuild;
  private final Predicate<PathFragment> allowedMissingInputs;
  private final boolean errorOnExternalFiles;

  private final ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions;
  private final ImmutableList<PrecomputedValue.Injected> extraPrecomputedValues;

  protected SkyframeIncrementalBuildMonitor incrementalBuildMonitor =
      new SkyframeIncrementalBuildMonitor();

  private MutableSupplier<ConfigurationFactory> configurationFactory = new MutableSupplier<>();
  private MutableSupplier<ImmutableList<ConfigurationFragmentFactory>> configurationFragments =
      new MutableSupplier<>();
  private MutableSupplier<Set<Package>> configurationPackages = new MutableSupplier<>();

  private static final Logger LOG = Logger.getLogger(SkyframeExecutor.class.getName());

  protected SkyframeExecutor(
      Reporter reporter,
      EvaluatorSupplier evaluatorSupplier,
      PackageFactory pkgFactory,
      TimestampGranularityMonitor tsgm,
      BlazeDirectories directories,
      Factory workspaceStatusActionFactory,
      ImmutableList<BuildInfoFactory> buildInfoFactories,
      Set<Path> immutableDirectories,
      Predicate<PathFragment> allowedMissingInputs,
      Preprocessor.Factory.Supplier preprocessorFactorySupplier,
      ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions,
      ImmutableList<PrecomputedValue.Injected> extraPrecomputedValues,
      boolean errorOnExternalFiles) {
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
    this.errorEventListener = this.reporter;
    this.resourceManager = ResourceManager.instance();
    this.skyframeActionExecutor = new SkyframeActionExecutor(reporter, resourceManager, eventBus,
        statusReporterRef);
    this.directories = Preconditions.checkNotNull(directories);
    this.buildInfoFactories = buildInfoFactories;
    this.immutableDirectories = immutableDirectories;
    this.allowedMissingInputs = allowedMissingInputs;
    this.preprocessorFactorySupplier = preprocessorFactorySupplier;
    this.extraSkyFunctions = extraSkyFunctions;
    this.extraPrecomputedValues = extraPrecomputedValues;
    this.errorOnExternalFiles = errorOnExternalFiles;
  }

  private ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions(
      Root buildDataDirectory,
      PackageFactory pkgFactory,
      Predicate<PathFragment> allowedMissingInputs) {
    ExternalFilesHelper externalFilesHelper = new ExternalFilesHelper(pkgLocator,
        immutableDirectories, errorOnExternalFiles);
    RuleClassProvider ruleClassProvider = pkgFactory.getRuleClassProvider();
    // We use an immutable map builder for the nice side effect that it throws if a duplicate key
    // is inserted.
    ImmutableMap.Builder<SkyFunctionName, SkyFunction> map = ImmutableMap.builder();
    map.put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction());
    map.put(SkyFunctions.FILE_STATE, new FileStateFunction(tsgm, externalFilesHelper));
    map.put(SkyFunctions.DIRECTORY_LISTING_STATE,
        new DirectoryListingStateFunction(externalFilesHelper));
    map.put(SkyFunctions.FILE_SYMLINK_CYCLE_UNIQUENESS,
        new FileSymlinkCycleUniquenessFunction());
    map.put(SkyFunctions.FILE_SYMLINK_INFINITE_EXPANSION_UNIQUENESS,
        new FileSymlinkInfiniteExpansionUniquenessFunction());
    map.put(SkyFunctions.FILE, new FileFunction(pkgLocator, tsgm, externalFilesHelper));
    map.put(SkyFunctions.DIRECTORY_LISTING, new DirectoryListingFunction());
    map.put(SkyFunctions.PACKAGE_LOOKUP, new PackageLookupFunction(deletedPackages));
    map.put(SkyFunctions.CONTAINING_PACKAGE_LOOKUP, new ContainingPackageLookupFunction());
    map.put(SkyFunctions.AST_FILE_LOOKUP, new ASTFileLookupFunction(
        pkgLocator, packageManager, ruleClassProvider));
    map.put(SkyFunctions.SKYLARK_IMPORTS_LOOKUP, new SkylarkImportLookupFunction(
        ruleClassProvider, pkgFactory));
    map.put(SkyFunctions.GLOB, newGlobFunction());
    map.put(SkyFunctions.TARGET_PATTERN, new TargetPatternFunction(pkgLocator));
    map.put(SkyFunctions.PREPARE_DEPS_OF_PATTERNS, new PrepareDepsOfPatternsFunction());
    map.put(SkyFunctions.PREPARE_DEPS_OF_PATTERN, new PrepareDepsOfPatternFunction(pkgLocator));
    map.put(SkyFunctions.PREPARE_DEPS_OF_TARGETS_UNDER_DIRECTORY,
        new PrepareDepsOfTargetsUnderDirectoryFunction());
    map.put(SkyFunctions.RECURSIVE_PKG, new RecursivePkgFunction());
    map.put(SkyFunctions.PACKAGE, new PackageFunction(
        reporter, pkgFactory, packageManager, showLoadingProgress, packageFunctionCache,
        numPackagesLoaded));
    map.put(SkyFunctions.TARGET_MARKER, new TargetMarkerFunction());
    map.put(SkyFunctions.TRANSITIVE_TARGET, new TransitiveTargetFunction(ruleClassProvider));
    map.put(SkyFunctions.TRANSITIVE_TRAVERSAL, new TransitiveTraversalFunction());
    map.put(SkyFunctions.CONFIGURED_TARGET,
        new ConfiguredTargetFunction(new BuildViewProvider(), ruleClassProvider));
    map.put(SkyFunctions.ASPECT, new AspectFunction(new BuildViewProvider(), ruleClassProvider));
    map.put(SkyFunctions.POST_CONFIGURED_TARGET,
        new PostConfiguredTargetFunction(new BuildViewProvider(), ruleClassProvider));
    map.put(SkyFunctions.BUILD_CONFIGURATION,
        new BuildConfigurationFunction(directories, ruleClassProvider));
    map.put(SkyFunctions.CONFIGURATION_COLLECTION, new ConfigurationCollectionFunction(
        configurationFactory, configurationPackages));
    map.put(SkyFunctions.CONFIGURATION_FRAGMENT, new ConfigurationFragmentFunction(
        configurationFragments, configurationPackages));
    map.put(
        SkyFunctions.WORKSPACE_FILE,
        new WorkspaceFileFunction(ruleClassProvider, pkgFactory, directories));
    map.put(SkyFunctions.TARGET_COMPLETION, CompletionFunction.targetCompletionFunction(eventBus));
    map.put(SkyFunctions.ASPECT_COMPLETION, CompletionFunction.aspectCompletionFunction(eventBus));
    map.put(SkyFunctions.TEST_COMPLETION, new TestCompletionFunction());
    map.put(SkyFunctions.ARTIFACT, new ArtifactFunction(allowedMissingInputs));
    map.put(SkyFunctions.BUILD_INFO_COLLECTION, new BuildInfoCollectionFunction(artifactFactory,
        buildDataDirectory));
    map.put(SkyFunctions.BUILD_INFO, new WorkspaceStatusFunction());
    map.put(SkyFunctions.COVERAGE_REPORT, new CoverageReportFunction());
    ActionExecutionFunction actionExecutionFunction =
        new ActionExecutionFunction(skyframeActionExecutor, tsgm);
    map.put(SkyFunctions.ACTION_EXECUTION, actionExecutionFunction);
    this.actionExecutionFunction = actionExecutionFunction;
    map.put(SkyFunctions.RECURSIVE_FILESYSTEM_TRAVERSAL,
        new RecursiveFilesystemTraversalFunction());
    map.put(SkyFunctions.FILESET_ENTRY, new FilesetEntryFunction());
    map.putAll(extraSkyFunctions);
    return map.build();
  }

  protected SkyFunction newGlobFunction() {
    return new GlobFunction(/*alwaysUseDirListing=*/false);
  }

  protected PerBuildSyscallCache newPerBuildSyscallCache() {
    return PerBuildSyscallCache.newUnboundedCache();
  }

 @ThreadCompatible
  public void setActive(boolean active) {
    this.active = active;
  }

  protected void checkActive() {
    Preconditions.checkState(active);
  }

  public void setFileCache(ActionInputFileCache fileCache) {
    this.skyframeActionExecutor.setFileCache(fileCache);
  }

  public void dump(boolean summarize, PrintStream out) {
    memoizingEvaluator.dump(summarize, out);
  }

  public abstract void dumpPackages(PrintStream out);

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

  /**
   * This method exists only to allow a module to make a top-level Skyframe call during the
   * transition to making it fully Skyframe-compatible. Do not add additional callers!
   */
  public <E extends Exception> SkyValue evaluateSkyKeyForCodeMigration(final SkyKey key,
      final Class<E> clazz) throws E {
    try {
      return callUninterruptibly(new Callable<SkyValue>() {
        @Override
        public SkyValue call() throws E, InterruptedException {
          synchronized (valueLookupLock) {
            // We evaluate in keepGoing mode because in the case that the graph does not store its
            // edges, nokeepGoing builds are not allowed, whereas keepGoing builds are always
            // permitted.
            EvaluationResult<SkyValue> result = buildDriver.evaluate(
                ImmutableList.of(key), true, ResourceUsage.getAvailableProcessors(),
                errorEventListener);
            if (!result.hasError()) {
              return Preconditions.checkNotNull(result.get(key), "%s %s", result, key);
            }
            ErrorInfo errorInfo = Preconditions.checkNotNull(result.getError(key),
                "%s %s", key, result);
            Throwables.propagateIfPossible(errorInfo.getException(), clazz);
            if (errorInfo.getException() != null) {
              throw new IllegalStateException(errorInfo.getException());
            }
            throw new IllegalStateException(errorInfo.toString());
          }
        }
      });
    } catch (Exception e) {
      Throwables.propagateIfPossible(e, clazz);
      throw new IllegalStateException(e);
    }
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
   * Must be called before the {@link SkyframeExecutor} can be used (should only be called in
   * factory methods and as an implementation detail of {@link #resetEvaluator}).
   */
  protected void init() {
    progressReceiver = newSkyframeProgressReceiver();
    Map<SkyFunctionName, SkyFunction> skyFunctions = skyFunctions(
        directories.getBuildDataDirectory(), pkgFactory, allowedMissingInputs);
    memoizingEvaluator = evaluatorSupplier.create(
        skyFunctions, evaluatorDiffer(), progressReceiver, emittedEventState,
        hasIncrementalState());
    buildDriver = newBuildDriver();
  }

  protected SkyframeProgressReceiver newSkyframeProgressReceiver() {
    return new SkyframeProgressReceiver();
  }

  /**
   * Reinitializes the Skyframe evaluator, dropping all previously computed values.
   *
   * <p>Be careful with this method as it also deletes all injected values. You need to make sure
   * that any necessary precomputed values are reinjected before the next build. Constants can be
   * put in {@link #reinjectConstantValuesLazily}.
   */
  public void resetEvaluator() {
    init();
    emittedEventState.clear();
    if (skyframeBuildView != null) {
      skyframeBuildView.clearLegacyData();
    }
    reinjectConstantValuesLazily();
  }

  protected abstract Differencer evaluatorDiffer();

  protected abstract BuildDriver newBuildDriver();

  /**
   * Values whose values are known at startup and guaranteed constant are still wiped from the
   * evaluator when we create a new one, so they must be re-injected each time we create a new
   * evaluator.
   */
  private void reinjectConstantValuesLazily() {
    needToInjectEmbeddedArtifacts = true;
    needToInjectPrecomputedValuesForAnalysis = true;
  }

  /**
   * Deletes all ConfiguredTarget values from the Skyframe cache. This is done to save memory (e.g.
   * on a configuration change); since the configuration is part of the key, these key/value pairs
   * will be sitting around doing nothing until the configuration changes back to the previous
   * value.
   *
   * <p>The next evaluation will delete all invalid values.
   */
  public abstract void dropConfiguredTargets();

  /**
   * Removes ConfigurationFragmentValuess and ConfigurationCollectionValues from the cache.
   */
  @VisibleForTesting
  public void invalidateConfigurationCollection() {
    invalidate(SkyFunctionName.functionIsIn(ImmutableSet.of(SkyFunctions.CONFIGURATION_FRAGMENT,
            SkyFunctions.CONFIGURATION_COLLECTION)));
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
    // Assume incrementality.
  }

  public boolean hasIncrementalState() {
    return true;
  }

  @VisibleForTesting
  protected abstract Injectable injectable();

  /**
   * Saves memory by clearing analysis objects from Skyframe. If using legacy execution, actually
   * deletes the relevant values. If using Skyframe execution, clears their data without deleting
   * them (they will be deleted on the next build).
   */
  public abstract void clearAnalysisCache(Collection<ConfiguredTarget> topLevelTargets);

  /**
   * Injects the contents of the computed tools/defaults package.
   */
  @VisibleForTesting
  public void setupDefaultPackage(String defaultsPackageContents) {
    PrecomputedValue.DEFAULTS_PACKAGE_CONTENTS.set(injectable(), defaultsPackageContents);
  }

  /**
   * Injects the top-level artifact options.
   */
  public void injectTopLevelContext(TopLevelArtifactContext options) {
    PrecomputedValue.TOP_LEVEL_CONTEXT.set(injectable(), options);
  }

  public void injectWorkspaceStatusData(BuildConfigurationCollection configurations) {
    PrecomputedValue.WORKSPACE_STATUS_KEY.set(injectable(),
        workspaceStatusActionFactory.createWorkspaceStatusAction(
            artifactFactory.get(), WorkspaceStatusValue.ARTIFACT_OWNER, buildId));
  }

  public void injectCoverageReportData(ImmutableList<Action> actions) {
    PrecomputedValue.COVERAGE_REPORT_KEY.set(injectable(), actions);
  }

  /**
   * Sets the default visibility.
   */
  private void setDefaultVisibility(RuleVisibility defaultVisibility) {
    PrecomputedValue.DEFAULT_VISIBILITY.set(injectable(), defaultVisibility);
  }

  private void maybeInjectPrecomputedValuesForAnalysis() {
    if (needToInjectPrecomputedValuesForAnalysis) {
      injectBuildInfoFactories();
      injectExtraPrecomputedValues();
      needToInjectPrecomputedValuesForAnalysis = false;
    }
  }

  private void injectExtraPrecomputedValues() {
    for (PrecomputedValue.Injected injected : extraPrecomputedValues) {
      injected.inject(injectable());
    }
  }

  protected Cache<PackageIdentifier, Package.LegacyBuilder> newPkgFunctionCache() {
    return CacheBuilder.newBuilder().build();
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
    PrecomputedValue.BUILD_INFO_FACTORIES.set(injectable(), factoryMapBuilder.build());
  }

  private void setShowLoadingProgress(boolean showLoadingProgressValue) {
    showLoadingProgress.set(showLoadingProgressValue);
  }

  @VisibleForTesting
  public void setCommandId(UUID commandId) {
    PrecomputedValue.BUILD_ID.set(injectable(), commandId);
    buildId.set(commandId);
  }

  /** Returns the build-info.txt and build-changelist.txt artifacts. */
  public Collection<Artifact> getWorkspaceStatusArtifacts() throws InterruptedException {
    // Should already be present, unless the user didn't request any targets for analysis.
    EvaluationResult<WorkspaceStatusValue> result = buildDriver.evaluate(
        ImmutableList.of(WorkspaceStatusValue.SKY_KEY), /*keepGoing=*/true, /*numThreads=*/1,
        reporter);
    WorkspaceStatusValue value =
        Preconditions.checkNotNull(result.get(WorkspaceStatusValue.SKY_KEY));
    return ImmutableList.of(value.getStableArtifact(), value.getVolatileArtifact());
  }

  // TODO(bazel-team): Make this take a PackageIdentifier.
  public Map<PathFragment, Root> getArtifactRoots(Iterable<PathFragment> execPaths)
      throws PackageRootResolutionException {
    final List<SkyKey> packageKeys = new ArrayList<>();
    for (PathFragment execPath : execPaths) {
      Preconditions.checkArgument(!execPath.isAbsolute(), execPath);
      packageKeys.add(ContainingPackageLookupValue.key(
          PackageIdentifier.createInDefaultRepo(execPath)));
    }

    EvaluationResult<ContainingPackageLookupValue> result;
    try {
      result = callUninterruptibly(new Callable<EvaluationResult<ContainingPackageLookupValue>>() {
        @Override
        public EvaluationResult<ContainingPackageLookupValue> call() throws InterruptedException {
          synchronized (valueLookupLock) {
            return buildDriver.evaluate(
                packageKeys, /*keepGoing=*/true, /*numThreads=*/1, reporter);
          }
        }
      });
    } catch (Exception e) {
      throw new IllegalStateException(e);  // Should never happen.
    }

    if (result.hasError()) {
      throw new PackageRootResolutionException("Exception encountered determining package roots",
          result.getError().getException());
    }

    Map<PathFragment, Root> roots = new HashMap<>();
    for (PathFragment execPath : execPaths) {
      ContainingPackageLookupValue value = result.get(ContainingPackageLookupValue.key(
          PackageIdentifier.createInDefaultRepo(execPath)));
      if (value.hasContainingPackage()) {
        roots.put(execPath, Root.asSourceRoot(value.getContainingPackageRoot()));
      } else {
        roots.put(execPath, null);
      }
    }
    return roots;
  }

  @VisibleForTesting
  public WorkspaceStatusAction getLastWorkspaceStatusActionForTesting() {
    PrecomputedValue value = (PrecomputedValue) buildDriver.getGraphForTesting()
        .getExistingValueForTesting(PrecomputedValue.WORKSPACE_STATUS_KEY.getKeyForTesting());
    return (WorkspaceStatusAction) value.get();
  }

  /**
   * Informs user about number of modified files (source and output files).
   */
  // Note, that number of modified files in some cases can be bigger than actual number of
  // modified files for targets in current request. Skyframe may check for modification all files
  // from previous requests.
  protected void informAboutNumberOfModifiedFiles() {
    LOG.info(String.format("Found %d modified files from last build", modifiedFiles));
  }

  public Reporter getReporter() {
    return reporter;
  }

  public EventBus getEventBus() {
    return eventBus.get();
  }

  public ActionExecutionContextFactory getActionExecutionContextFactory() {
    return skyframeActionExecutor;
  }

  /**
   * The map from package names to the package root where each package was found; this is used to
   * set up the symlink tree.
   */
  public ImmutableMap<PackageIdentifier, Path> getPackageRoots() {
    // Make a map of the package names to their root paths.
    ImmutableMap.Builder<PackageIdentifier, Path> packageRoots = ImmutableMap.builder();
    for (Package pkg : configurationPackages.get()) {
      packageRoots.put(pkg.getPackageIdentifier(), pkg.getSourceRoot());
    }
    return packageRoots.build();
  }

  @VisibleForTesting
  ImmutableList<Path> getPathEntries() {
    return pkgLocator.get().getPathEntries();
  }

  protected abstract void invalidate(Predicate<SkyKey> pred);

  private static boolean compatibleFileTypes(Dirent.Type oldType, FileStateValue.Type newType) {
    return (oldType.equals(Dirent.Type.FILE) && newType.equals(FileStateValue.Type.FILE))
        || (oldType.equals(Dirent.Type.DIRECTORY) && newType.equals(FileStateValue.Type.DIRECTORY))
        || (oldType.equals(Dirent.Type.SYMLINK) && newType.equals(FileStateValue.Type.SYMLINK));
  }

  protected Differencer.Diff getDiff(Iterable<PathFragment> modifiedSourceFiles,
      final Path pathEntry) throws InterruptedException {
    // TODO(bazel-team): change ModifiedFileSet to work with RootedPaths instead of PathFragments.
    Iterable<SkyKey> dirtyFileStateSkyKeys = Iterables.transform(modifiedSourceFiles,
        new Function<PathFragment, SkyKey>() {
          @Override
          public SkyKey apply(PathFragment pathFragment) {
            Preconditions.checkState(!pathFragment.isAbsolute(),
                "found absolute PathFragment: %s", pathFragment);
            return FileStateValue.key(RootedPath.toRootedPath(pathEntry, pathFragment));
          }
        });
    // We only need to invalidate directory values when a file has been created or deleted or
    // changes type, not when it has merely been modified. Unfortunately we do not have that
    // information here, so we compute it ourselves.
    // TODO(bazel-team): Fancy filesystems could provide it with a hypothetically modified
    // DiffAwareness interface.
    Supplier<Map<SkyKey, SkyValue>> valuesSupplier = Suppliers.memoize(
        new Supplier<Map<SkyKey, SkyValue>>() {
      @Override
      public Map<SkyKey, SkyValue> get() {
        return memoizingEvaluator.getValues();
      }
    });
    FilesystemValueChecker fsvc = new FilesystemValueChecker(valuesSupplier, tsgm, null);
    Differencer.DiffWithDelta diff =
        fsvc.getNewAndOldValues(dirtyFileStateSkyKeys, new FileDirtinessChecker());

    Set<SkyKey> valuesToInvalidate = new HashSet<>();
    Map<SkyKey, SkyValue> valuesToInject = new HashMap<>();
    for (Map.Entry<SkyKey, Delta> entry : diff.changedKeysWithNewAndOldValues().entrySet()) {
      SkyKey key = entry.getKey();
      Preconditions.checkState(key.functionName().equals(SkyFunctions.FILE_STATE), key);
      RootedPath rootedPath = (RootedPath) key.argument();
      Delta delta = entry.getValue();
      FileStateValue oldValue = (FileStateValue) delta.getOldValue();
      FileStateValue newValue = (FileStateValue) delta.getNewValue();
      if (newValue != null) {
        valuesToInject.put(key, newValue);
      } else {
        valuesToInvalidate.add(key);
      }
      SkyKey dirListingStateKey = parentDirectoryListingStateKey(rootedPath);
      // Invalidate the directory listing for the path's parent directory if the change was
      // relevant (e.g. path turned from a symlink into a directory) OR if we don't have enough
      // information to determine it was irrelevant.
      boolean changedType = false;
      if (newValue == null) {
        changedType = true;
      } else if (oldValue != null) {
        changedType = !oldValue.getType().equals(newValue.getType());
      } else {
        DirectoryListingStateValue oldDirListingStateValue =
            (DirectoryListingStateValue) valuesSupplier.get().get(dirListingStateKey);
        if (oldDirListingStateValue != null) {
          String baseName = rootedPath.getRelativePath().getBaseName();
          Dirent oldDirent = oldDirListingStateValue.getDirents().maybeGetDirent(baseName);
          changedType = (oldDirent == null)
              || !compatibleFileTypes(oldDirent.getType(), newValue.getType());
        } else {
          changedType = true;
        }
      }
      if (changedType) {
        valuesToInvalidate.add(dirListingStateKey);
      }
    }
    for (SkyKey key : diff.changedKeysWithoutNewValues()) {
      Preconditions.checkState(key.functionName().equals(SkyFunctions.FILE_STATE), key);
      RootedPath rootedPath = (RootedPath) key.argument();
      valuesToInvalidate.add(parentDirectoryListingStateKey(rootedPath));
    }
    return new ImmutableDiff(valuesToInvalidate, valuesToInject);
  }

  private static SkyKey parentDirectoryListingStateKey(RootedPath rootedPath) {
    RootedPath parentDirRootedPath = RootedPath.toRootedPath(
        rootedPath.getRoot(), rootedPath.getRelativePath().getParentDirectory());
    return DirectoryListingStateValue.key(parentDirRootedPath);
  }

  protected static SkyKey createFileStateKey(RootedPath rootedPath) {
    return FileStateValue.key(rootedPath);
  }

  protected static SkyKey createDirectoryListingStateKey(RootedPath rootedPath) {
    return DirectoryListingStateValue.key(rootedPath);
  }

  /**
   * Creates a FileValue pointing of type directory. No matter that the rootedPath points to a
   * symlink.
   *
   * <p> Use it with caution as it would prevent invalidation when the destination file in the
   * symlink changes.
   */
  protected static FileValue createFileDirValue(RootedPath rootedPath) {
    return FileValue.value(rootedPath, FileStateValue.DIRECTORY_FILE_STATE_NODE,
        rootedPath, FileStateValue.DIRECTORY_FILE_STATE_NODE);
  }

  /**
   * Sets the packages that should be treated as deleted and ignored.
   */
  @VisibleForTesting  // productionVisibility = Visibility.PRIVATE
  public abstract void setDeletedPackages(Iterable<PackageIdentifier> pkgs);

  /**
   * Prepares the evaluator for loading.
   *
   * <p>MUST be run before every incremental build.
   */
  @VisibleForTesting  // productionVisibility = Visibility.PRIVATE
  public void preparePackageLoading(PathPackageLocator pkgLocator, RuleVisibility defaultVisibility,
                                    boolean showLoadingProgress, int globbingThreads,
                                    String defaultsPackageContents, UUID commandId) {
    Preconditions.checkNotNull(pkgLocator);
    setActive(true);

    maybeInjectPrecomputedValuesForAnalysis();
    setCommandId(commandId);
    setShowLoadingProgress(showLoadingProgress);
    setDefaultVisibility(defaultVisibility);
    setupDefaultPackage(defaultsPackageContents);
    setPackageLocator(pkgLocator);

    syscalls.set(newPerBuildSyscallCache());
    this.pkgFactory.setGlobbingThreads(globbingThreads);
    checkPreprocessorFactory();
    emittedEventState.clear();

    // If the PackageFunction was interrupted, there may be stale entries here.
    packageFunctionCache.invalidateAll();
    numPackagesLoaded.set(0);

    // Reset the stateful SkyframeCycleReporter, which contains cycles from last run.
    cyclesReporter.set(createCyclesReporter());
  }

  @SuppressWarnings("unchecked")
  private void setPackageLocator(PathPackageLocator pkgLocator) {
    PathPackageLocator oldLocator = this.pkgLocator.getAndSet(pkgLocator);
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(injectable(), pkgLocator);

    if (!pkgLocator.equals(oldLocator)) {
      // The package path is read not only by SkyFunctions but also by some other code paths.
      // We need to take additional steps to keep the corresponding data structures in sync.
      // (Some of the additional steps are carried out by ConfiguredTargetValueInvalidationListener,
      // and some by BuildView#buildHasIncompatiblePackageRoots and #updateSkyframe.)
      onNewPackageLocator(oldLocator, pkgLocator);
    }
  }

  protected abstract void onNewPackageLocator(PathPackageLocator oldLocator,
                                              PathPackageLocator pkgLocator);

  private void checkPreprocessorFactory() {
    if (preprocessorFactory == null) {
      Preprocessor.Factory newPreprocessorFactory = preprocessorFactorySupplier.getFactory(
          packageManager);
      pkgFactory.setPreprocessorFactory(newPreprocessorFactory);
      preprocessorFactory = newPreprocessorFactory;
    } else if (!preprocessorFactory.isStillValid()) {
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
    this.artifactFactory.set(skyframeBuildView.getArtifactFactory());
  }

  /**
   * Sets the eventBus to use for posting events.
   */
  public void setEventBus(EventBus eventBus) {
    this.eventBus.set(eventBus);
  }

  /**
   * Sets the path for action log buffers.
   */
  public void setActionOutputRoot(Path actionOutputRoot) {
    Preconditions.checkNotNull(actionOutputRoot);
    this.actionLogBufferPathGenerator = new ActionLogBufferPathGenerator(actionOutputRoot);
    this.skyframeActionExecutor.setActionLogBufferPathGenerator(actionLogBufferPathGenerator);
  }

  @VisibleForTesting
  public void setConfigurationDataForTesting(BlazeDirectories directories,
      ConfigurationFactory configurationFactory) {
    PrecomputedValue.BLAZE_DIRECTORIES.set(injectable(), directories);
    this.configurationFactory.set(configurationFactory);
    this.configurationFragments.set(ImmutableList.copyOf(configurationFactory.getFactories()));
    this.configurationPackages.set(Sets.<Package>newConcurrentHashSet());
  }

  /**
   * Asks the Skyframe evaluator to build the value for BuildConfigurationCollection and returns the
   * result. Also invalidates {@link PrecomputedValue#BLAZE_DIRECTORIES} if it has changed.
   */
  public BuildConfigurationCollection createConfigurations(
      ConfigurationFactory configurationFactory, BuildOptions buildOptions,
      BlazeDirectories directories, Set<String> multiCpu, boolean keepGoing)
      throws InvalidConfigurationException, InterruptedException {
    this.configurationPackages.set(Sets.<Package>newConcurrentHashSet());
    this.configurationFactory.set(configurationFactory);
    this.configurationFragments.set(ImmutableList.copyOf(configurationFactory.getFactories()));
    // TODO(bazel-team): find a way to use only BuildConfigurationKey instead of
    // BlazeDirectories.
    PrecomputedValue.BLAZE_DIRECTORIES.set(injectable(), directories);

    SkyKey skyKey = ConfigurationCollectionValue.key(
        buildOptions, ImmutableSortedSet.copyOf(multiCpu));
    EvaluationResult<ConfigurationCollectionValue> result = buildDriver.evaluate(
            Arrays.asList(skyKey), keepGoing, DEFAULT_THREAD_COUNT, errorEventListener);
    if (result.hasError()) {
      Throwable e = result.getError(skyKey).getException();
      // Wrap loading failed exceptions
      if (e instanceof NoSuchThingException) {
        e = new InvalidConfigurationException(e);
      }
      Throwables.propagateIfInstanceOf(e, InvalidConfigurationException.class);
      throw new IllegalStateException(
          "Unknown error during ConfigurationCollectionValue evaluation", e);
    }
    Preconditions.checkState(result.values().size() == 1,
        "Result of evaluate() must contain exactly one value %s", result);
    ConfigurationCollectionValue configurationValue =
        Iterables.getOnlyElement(result.values());
    this.configurationPackages.set(
        Sets.newConcurrentHashSet(configurationValue.getConfigurationPackages()));
    return configurationValue.getConfigurationCollection();
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
  ImmutableMap<Action, SkyframeActionExecutor.ConflictException> findArtifactConflicts()
      throws InterruptedException {
    if (skyframeBuildView.isSomeConfiguredTargetEvaluated()
        || skyframeBuildView.isSomeConfiguredTargetInvalidated()) {
      // This operation is somewhat expensive, so we only do it if the graph might have changed in
      // some way -- either we analyzed a new target or we invalidated an old one.
      long startTime = Profiler.nanoTimeMaybe();
      skyframeActionExecutor.findAndStoreArtifactConflicts(getActionLookupValues());
      skyframeBuildView.resetEvaluatedConfiguredTargetFlag();
      // The invalidated configured targets flag will be reset later in the evaluate() call.

      long duration = Profiler.nanoTimeMaybe() - startTime;
      if (duration > 0) {
        LOG.info("Spent " + (duration / 1000 / 1000) + " ms discovering artifact conflicts");
      }
    }
    return skyframeActionExecutor.badActions();
  }

  /**
   * Asks the Skyframe evaluator to build the given artifacts and targets, and to test the
   * given test targets.
   */
  public EvaluationResult<?> buildArtifacts(
      Executor executor,
      Set<Artifact> artifactsToBuild,
      Collection<ConfiguredTarget> targetsToBuild,
      Collection<AspectValue> aspects,
      Collection<ConfiguredTarget> targetsToTest,
      boolean exclusiveTesting,
      boolean keepGoing,
      boolean explain,
      int numJobs,
      ActionCacheChecker actionCacheChecker,
      @Nullable EvaluationProgressReceiver executionProgressReceiver)
      throws InterruptedException {
    checkActive();
    Preconditions.checkState(actionLogBufferPathGenerator != null);

    skyframeActionExecutor.prepareForExecution(executor, keepGoing, explain, actionCacheChecker);

    resourceManager.resetResourceUsage();
    try {
      progressReceiver.executionProgressReceiver = executionProgressReceiver;
      Iterable<SkyKey> artifactKeys = ArtifactValue.mandatoryKeys(artifactsToBuild);
      Iterable<SkyKey> targetKeys = TargetCompletionValue.keys(targetsToBuild);
      Iterable<SkyKey> aspectKeys = AspectCompletionValue.keys(aspects);
      Iterable<SkyKey> testKeys = TestCompletionValue.keys(targetsToTest, exclusiveTesting);
      return buildDriver.evaluate(
          Iterables.concat(artifactKeys, targetKeys, aspectKeys, testKeys),
          keepGoing,
          numJobs,
          errorEventListener);
    } finally {
      progressReceiver.executionProgressReceiver = null;
      // Also releases thread locks.
      resourceManager.resetResourceUsage();
      skyframeActionExecutor.executionOver();
      actionExecutionFunction.complete();
    }
  }

  @VisibleForTesting
  public void prepareBuildingForTestingOnly(Executor executor, boolean keepGoing, boolean explain,
                                            ActionCacheChecker checker) {
    skyframeActionExecutor.prepareForExecution(executor, keepGoing, explain, checker);
  }

  EvaluationResult<TargetPatternValue> targetPatterns(Iterable<SkyKey> patternSkyKeys,
      int numThreads, boolean keepGoing, EventHandler eventHandler) throws InterruptedException {
    checkActive();
    return buildDriver.evaluate(patternSkyKeys, keepGoing, numThreads, eventHandler);
  }

  /**
   * Returns the {@link ConfiguredTarget}s corresponding to the given keys.
   *
   * <p>For use for legacy support from {@code BuildView} only.
   *
   * <p>If a requested configured target is in error, the corresponding value is omitted from the
   * returned list.
   */
  @ThreadSafety.ThreadSafe
  public ImmutableList<ConfiguredTarget> getConfiguredTargets(BuildConfiguration originalConfig,
      Iterable<Dependency> keys, boolean useOriginalConfig) {
    return getConfiguredTargetMap(originalConfig, keys, useOriginalConfig).values().asList();
  }

  @ThreadSafety.ThreadSafe
  public ImmutableMap<Dependency, ConfiguredTarget> getConfiguredTargetMap(
      BuildConfiguration originalConfig, Iterable<Dependency> keys, boolean useOriginalConfig) {
    checkActive();
    if (skyframeBuildView == null) {
      // If build view has not yet been initialized, no configured targets can have been created.
      // This is most likely to happen after a failed loading phase.
      return ImmutableMap.of();
    }

    Map<Dependency, BuildConfiguration> configs;
    if (originalConfig != null) {
      if (useOriginalConfig) {
        // This flag is used because of some unfortunate complexity in the configuration machinery:
        // Most callers of this method pass a <Label, Configuration> pair to directly create a
        // ConfiguredTarget from, but happen to use the Dependency data structure to pass that
        // info (even though the data has nothing to do with dependencies). If this configuration
        // includes a split transition, a dynamic configuration created from it will *not*
        // include that transition (because dynamic configurations don't embed transitions to
        // other configurations. In that case, we need to preserve the original configuration.
        // TODO(bazel-team); make this unnecessary once split transition logic is properly ported
        // out of configurations.
        configs = new HashMap<>();
        configs.put(Iterables.getOnlyElement(keys), originalConfig);
      } else {
        configs = getConfigurations(originalConfig.getOptions(), keys);
      }
    } else {
      configs = new HashMap<>();
      for (Dependency key : keys) {
        configs.put(key, null);
      }
    }

    final List<SkyKey> skyKeys = new ArrayList<>();
    for (Dependency key : keys) {
      skyKeys.add(ConfiguredTargetValue.key(key.getLabel(), configs.get(key)));
      for (AspectWithParameters aspect : key.getAspects()) {
        skyKeys.add(
            ConfiguredTargetFunction.createAspectKey(key.getLabel(), configs.get(key), aspect));
      }
    }

    EvaluationResult<SkyValue> result = evaluateSkyKeys(skyKeys);
    ImmutableMap.Builder<Dependency, ConfiguredTarget> cts = ImmutableMap.builder();

  DependentNodeLoop:
    for (Dependency key : keys) {
      SkyKey configuredTargetKey = ConfiguredTargetValue.key(
          key.getLabel(), configs.get(key));
      if (result.get(configuredTargetKey) == null) {
        continue;
      }

      ConfiguredTarget configuredTarget =
          ((ConfiguredTargetValue) result.get(configuredTargetKey)).getConfiguredTarget();
      List<Aspect> aspects = new ArrayList<>();

      for (AspectWithParameters aspect : key.getAspects()) {
        SkyKey aspectKey =
            ConfiguredTargetFunction.createAspectKey(key.getLabel(), configs.get(key), aspect);
        if (result.get(aspectKey) == null) {
          continue DependentNodeLoop;
        }

        aspects.add(((AspectValue) result.get(aspectKey)).getAspect());
      }

      cts.put(key, RuleConfiguredTarget.mergeAspects(configuredTarget, aspects));
    }

    return cts.build();
  }

  /**
   * Retrieves the configurations needed for the given deps, trimming down their fragments
   * to those only needed by their transitive closures.
   */
  private Map<Dependency, BuildConfiguration> getConfigurations(BuildOptions fromOptions,
      Iterable<Dependency> keys) {
    Map<Dependency, BuildConfiguration> builder = new HashMap<>();
    Set<Dependency> depsToEvaluate = new HashSet<>();

    // Check: if !Configuration.useDynamicConfigs then just return the original configs.

    // Get the fragments needed for dynamic configuration nodes.
    final List<SkyKey> transitiveFragmentSkyKeys = new ArrayList<>();
    Map<Label, Set<Class<? extends BuildConfiguration.Fragment>>> fragmentsMap = new HashMap<>();
    Set<Label> labelsWithErrors = new HashSet<>();
    for (Dependency key : keys) {
      if (key.hasStaticConfiguration()) {
        builder.put(key, key.getConfiguration());
      } else if (key.getTransition() == Attribute.ConfigurationTransition.NULL) {
        builder.put(key, null);
      } else {
        depsToEvaluate.add(key);
        transitiveFragmentSkyKeys.add(TransitiveTargetValue.key(key.getLabel()));
      }
    }
    EvaluationResult<SkyValue> fragmentsResult = evaluateSkyKeys(transitiveFragmentSkyKeys);
    for (Dependency key : keys) {
      if (!depsToEvaluate.contains(key)) {
        // No fragments to compute here.
      } else if (fragmentsResult.getError(TransitiveTargetValue.key(key.getLabel())) != null) {
        labelsWithErrors.add(key.getLabel());
      } else {
        TransitiveTargetValue ttv =
            (TransitiveTargetValue) fragmentsResult.get(TransitiveTargetValue.key(key.getLabel()));
        fragmentsMap.put(key.getLabel(), ttv.getTransitiveConfigFragments().toSet());
      }
    }

    // Now get the configurations.
    final List<SkyKey> configSkyKeys = new ArrayList<>();
    for (Dependency key : keys) {
      if (!depsToEvaluate.contains(key) || labelsWithErrors.contains(key.getLabel())) {
        continue;
      }
      configSkyKeys.add(BuildConfigurationValue.key(fragmentsMap.get(key.getLabel()),
          getDynamicConfigOptions(key, fromOptions)));
    }
    EvaluationResult<SkyValue> configsResult = evaluateSkyKeys(configSkyKeys);
    for (Dependency key : keys) {
      if (!depsToEvaluate.contains(key) || labelsWithErrors.contains(key.getLabel())) {
        continue;
      }
      SkyKey configKey = BuildConfigurationValue.key(fragmentsMap.get(key.getLabel()),
          getDynamicConfigOptions(key, fromOptions));
      builder.put(key, ((BuildConfigurationValue) configsResult.get(configKey)).getConfiguration());
    }

    return builder;
  }

  /**
   * Computes the build options needed for the given key, accounting for transitions possibly
   * specified in the key.
   */
  private BuildOptions getDynamicConfigOptions(Dependency key, BuildOptions fromOptions) {
    if (key.hasStaticConfiguration()) {
      return key.getConfiguration().getOptions();
    } else if (key.getTransition() == Attribute.ConfigurationTransition.NONE) {
      return fromOptions;
    } else {
      return ((PatchTransition) key.getTransition()).apply(fromOptions);
    }
  }

  /**
   * Evaluates the given sky keys, blocks, and returns their evaluation results.
   */
  private EvaluationResult<SkyValue> evaluateSkyKeys(final Iterable<SkyKey> skyKeys) {
    EvaluationResult<SkyValue> result;
    try {
      result = callUninterruptibly(new Callable<EvaluationResult<SkyValue>>() {
        @Override
        public EvaluationResult<SkyValue> call() throws Exception {
          synchronized (valueLookupLock) {
            try {
              skyframeBuildView.enableAnalysis(true);
              return buildDriver.evaluate(skyKeys, false, DEFAULT_THREAD_COUNT, errorEventListener);
            } finally {
              skyframeBuildView.enableAnalysis(false);
            }
          }
        }
      });
    } catch (Exception e) {
      throw new IllegalStateException(e);  // Should never happen.
    }
    return result;
  }

  /**
   * Returns a dynamic configuration constructed from the given configuration fragments and build
   * options.
   */
  @VisibleForTesting
  public BuildConfiguration getConfigurationForTesting(
      Set<Class<? extends BuildConfiguration.Fragment>> fragments, BuildOptions options)
      throws InterruptedException {
    SkyKey key = BuildConfigurationValue.key(fragments, options);
    BuildConfigurationValue result = (BuildConfigurationValue) buildDriver
        .evaluate(ImmutableList.of(key), false, DEFAULT_THREAD_COUNT, errorEventListener).get(key);
    return result.getConfiguration();
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
        PrecomputedValue.WORKSPACE_STATUS_KEY.getKeyForTesting()) == null) {
      injectWorkspaceStatusData(null);
    }
    return Iterables.getFirst(
        getConfiguredTargets(configuration, ImmutableList.of(new Dependency(label, configuration)),
            true),
        null);
  }

  /**
   * Invalidates Skyframe values corresponding to the given set of modified files under the given
   * path entry.
   *
   * <p>May throw an {@link InterruptedException}, which means that no values have been invalidated.
   */
  @VisibleForTesting
  public abstract void invalidateFilesUnderPathForTesting(ModifiedFileSet modifiedFileSet,
      Path pathEntry) throws InterruptedException;

  /**
   * Invalidates SkyFrame values that may have failed for transient reasons.
   */
  public abstract void invalidateTransientErrors();

  @VisibleForTesting
  public TimestampGranularityMonitor getTimestampGranularityMonitorForTesting() {
    return tsgm;
  }

  /** Configures a given set of configured targets. */
  public EvaluationResult<ActionLookupValue> configureTargets(
      List<ConfiguredTargetKey> values, List<AspectKey> aspectKeys, boolean keepGoing)
      throws InterruptedException {
    checkActive();

    List<SkyKey> keys = new ArrayList<>(ConfiguredTargetValue.keys(values));
    for (AspectKey aspectKey : aspectKeys) {
      keys.add(AspectValue.key(aspectKey));
    }

    // Make sure to not run too many analysis threads. This can cause memory thrashing.
    return buildDriver.evaluate(
        keys, keepGoing, ResourceUsage.getAvailableProcessors(), errorEventListener);
  }

  /**
   * Post-process the targets. Values in the EvaluationResult are known to be transitively
   * error-free from action conflicts.
   */
  public EvaluationResult<PostConfiguredTargetValue> postConfigureTargets(
      List<ConfiguredTargetKey> values, boolean keepGoing,
      ImmutableMap<Action, SkyframeActionExecutor.ConflictException> badActions)
          throws InterruptedException {
    checkActive();
    PrecomputedValue.BAD_ACTIONS.set(injectable(), badActions);
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

    public Set<Package> retrievePackages(Set<PackageIdentifier> packageIds) {
      final List<SkyKey> valueNames = new ArrayList<>();
      for (PackageIdentifier pkgId : packageIds) {
        valueNames.add(PackageValue.key(pkgId));
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
   * For internal use in queries: performs a graph update to make sure the transitive closure of
   * the specified target {@code patterns} is present in the graph, and returns the {@link
   * EvaluationResult}.
   *
   * <p>The graph update is unconditionally done in keep-going mode, so that the query is guaranteed
   * a complete graph to work on.
   */
  @Override
  public EvaluationResult<SkyValue> prepareAndGet(Collection<String> patterns,
      int numThreads, EventHandler eventHandler) throws InterruptedException {
    SkyKey skyKey = getPrepareDepsKey(patterns);
    EvaluationResult<SkyValue> evaluationResult =
        buildDriver.evaluate(ImmutableList.of(skyKey), true, numThreads, eventHandler);
    Preconditions.checkNotNull(evaluationResult.getWalkableGraph(), patterns);
    return evaluationResult;
  }

  /**
   * Get metadata related to the prepareAndGet() lookup. Resulting data is specific to the
   * underlying evaluation implementation.
   */
  public String prepareAndGetMetadata(Collection<String> patterns) {
    return buildDriver.meta(ImmutableList.of(getPrepareDepsKey(patterns)));
  }

  private SkyKey getPrepareDepsKey(Collection<String> patterns) {
    SkyframeTargetPatternEvaluator patternEvaluator =
        (SkyframeTargetPatternEvaluator) packageManager.getTargetPatternEvaluator();
    String offset = patternEvaluator.getOffset();
    return PrepareDepsOfPatternsValue.key(ImmutableList.copyOf(patterns), offset);
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
     * Looks up a particular package (mostly used after the loading phase, so packages should
     * already be present, but occasionally used pre-loading phase). Use should be discouraged,
     * since this cannot be used inside a Skyframe evaluation, and concurrent calls are
     * synchronized.
     *
     * <p>Note that this method needs to be synchronized since InMemoryMemoizingEvaluator.evaluate()
     * method does not support concurrent calls.
     */
    Package getPackage(EventHandler eventHandler, PackageIdentifier pkgName)
        throws InterruptedException, NoSuchPackageException {
      synchronized (valueLookupLock) {
        SkyKey key = PackageValue.key(pkgName);
        // Any call to this method post-loading phase should either be error-free or be in a
        // keep_going build, since otherwise the build would have failed during loading. Thus
        // we set keepGoing=true unconditionally.
        EvaluationResult<PackageValue> result =
            buildDriver.evaluate(ImmutableList.of(key), /*keepGoing=*/true,
                DEFAULT_THREAD_COUNT, eventHandler);
        if (result.hasError()) {
          ErrorInfo error = result.getError();
          if (!Iterables.isEmpty(error.getCycleInfo())) {
            reportCycles(result.getError().getCycleInfo(), key);
            // This can only happen if a package is freshly loaded outside of the target parsing
            // or loading phase
            throw new BuildFileContainsErrorsException(
                pkgName, "Cycle encountered while loading package " + pkgName);
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

    Package getLoadedPackage(final PackageIdentifier pkgName) throws NoSuchPackageException {
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
    public boolean isPackageDeleted(PackageIdentifier packageName) {
      return deletedPackages.get().contains(packageName);
    }

    /** Same as {@link PackageManager#partiallyClear}. */
    void partiallyClear() {
      packageFunctionCache.invalidateAll();
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
  protected static final <T> T callUninterruptibly(Callable<T> callable) throws Exception {
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

  /**
   * Stores the set of loaded packages and, if needed, evicts ConfiguredTarget values.
   *
   * <p>The set represents all packages from the transitive closure of the top-level targets from
   * the latest build.
   */
  @ThreadCompatible
  public abstract void updateLoadedPackageSet(Set<PackageIdentifier> loadedPackages);

  public void sync(PackageCacheOptions packageCacheOptions, Path outputBase, Path workingDirectory,
      String defaultsPackageContents, UUID commandId) throws InterruptedException,
      AbruptExitException{

    preparePackageLoading(
        createPackageLocator(
            packageCacheOptions, outputBase, directories.getWorkspace(), workingDirectory),
        packageCacheOptions.defaultVisibility, packageCacheOptions.showLoadingProgress,
        packageCacheOptions.globbingThreads, defaultsPackageContents, commandId);
    setDeletedPackages(ImmutableSet.copyOf(packageCacheOptions.deletedPackages));

    incrementalBuildMonitor = new SkyframeIncrementalBuildMonitor();
    invalidateTransientErrors();
  }

  protected PathPackageLocator createPackageLocator(PackageCacheOptions packageCacheOptions,
      Path outputBase, Path workspace, Path workingDirectory) throws AbruptExitException {
    return PathPackageLocator.create(
        outputBase, packageCacheOptions.packagePath, getReporter(), workspace, workingDirectory);
  }

  private CyclesReporter createCyclesReporter() {
    return new CyclesReporter(
        new TransitiveTargetCycleReporter(packageManager),
        new ActionArtifactCycleReporter(packageManager),
        new SkylarkModuleCycleReporter(),
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
   * This should be called at most once in the lifetime of the SkyframeExecutor (except for
   * tests), and it should be called before the execution phase.
   */
  void setArtifactFactoryAndBinTools(ArtifactFactory artifactFactory, BinTools binTools) {
    this.artifactFactory.set(artifactFactory);
    this.binTools = binTools;
  }

  public void prepareExecution(boolean checkOutputFiles,
      @Nullable Range<Long> lastExecutionTimeRange) throws AbruptExitException,
      InterruptedException {
    maybeInjectEmbeddedArtifacts();

    if (checkOutputFiles) {
      // Detect external modifications in the output tree.
      FilesystemValueChecker fsvc = new FilesystemValueChecker(memoizingEvaluator, tsgm,
          lastExecutionTimeRange);
      invalidateDirtyActions(fsvc.getDirtyActionValues(batchStatter));
      modifiedFiles += fsvc.getNumberOfModifiedOutputFiles();
      outputDirtyFiles += fsvc.getNumberOfModifiedOutputFiles();
      modifiedFilesDuringPreviousBuild += fsvc.getNumberOfModifiedOutputFilesDuringPreviousBuild();
    }
    informAboutNumberOfModifiedFiles();
  }

  protected abstract void invalidateDirtyActions(Iterable<SkyKey> dirtyActionValues);

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
      FileArtifactValue fileArtifactValue;
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
    injectable().inject(values);
    needToInjectEmbeddedArtifacts = false;
  }

  /**
   * Mark dirty values for deletion if they've been dirty for longer than N versions.
   *
   * <p>Specifying a value N means, if the current version is V and a value was dirtied (and
   * has remained so) in version U, and U + N &lt;= V, then the value will be marked for deletion
   * and purged in version V+1.
   */
  public abstract void deleteOldNodes(long versionWindowForDirtyGc);

  /**
   * A progress received to track analysis invalidation and update progress messages.
   */
  protected class SkyframeProgressReceiver implements EvaluationProgressReceiver {
    /**
     * This flag is needed in order to avoid invalidating legacy data when we clear the
     * analysis cache because of --discard_analysis_cache flag. For that case we want to keep
     * the legacy data but get rid of the Skyframe data.
     */
    protected boolean ignoreInvalidations = false;
    /** This receiver is only needed for execution, so it is null otherwise. */
    @Nullable EvaluationProgressReceiver executionProgressReceiver = null;

    @Override
    public void invalidated(SkyKey skyKey, InvalidationState state) {
      if (ignoreInvalidations) {
        return;
      }
      if (skyframeBuildView != null) {
        skyframeBuildView.getInvalidationReceiver().invalidated(skyKey, state);
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
    public void computed(SkyKey skyKey, long elapsedTimeNanos) {}

    @Override
    public void evaluated(SkyKey skyKey, Supplier<SkyValue> valueSupplier, EvaluationState state) {
      if (ignoreInvalidations) {
        return;
      }
      if (skyframeBuildView != null) {
        skyframeBuildView.getInvalidationReceiver().evaluated(skyKey, valueSupplier, state);
      }
      if (executionProgressReceiver != null) {
        executionProgressReceiver.evaluated(skyKey, valueSupplier, state);
      }
    }
  }

  public int getOutputDirtyFilesAndClear() {
    int result = outputDirtyFiles;
    outputDirtyFiles = 0;
    return result;
  }

  public int getModifiedFilesDuringPreviousBuildAndClear() {
    int result = modifiedFilesDuringPreviousBuild;
    modifiedFilesDuringPreviousBuild = 0;
    return result;
  }
}
