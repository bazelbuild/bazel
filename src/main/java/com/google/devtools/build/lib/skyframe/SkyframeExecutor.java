// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.concurrent.Uninterruptibles.callUninterruptibly;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Stopwatch;
import com.google.common.base.Throwables;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionExecutionStatusReporter;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLogBufferPathGenerator;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ArtifactResolver.ArtifactResolverSupplier;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CompletionContext.PathResolverFactory;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.AnalysisProtos.ActionGraphContainer;
import com.google.devtools.build.lib.analysis.AspectCollection;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.Dependency;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.Factory;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsDiffForReconstruction;
import com.google.devtools.build.lib.analysis.config.ConfigurationResolver;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentClassSet;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.analysis.configuredtargets.MergedConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.MergedConfiguredTarget.DuplicateException;
import com.google.devtools.build.lib.analysis.skylark.StarlarkTransition;
import com.google.devtools.build.lib.analysis.skylark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.concurrent.NamedForkJoinPool;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.ErrorSensingEventHandler;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AstParseResult;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.TargetParsingPhaseTimeEvent;
import com.google.devtools.build.lib.pkgcache.TargetPatternPreloader;
import com.google.devtools.build.lib.pkgcache.TestFilter;
import com.google.devtools.build.lib.pkgcache.TransitivePackageLoader;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.rules.repository.ManagedDirectoriesKnowledge;
import com.google.devtools.build.lib.rules.repository.ResolvedFileFunction;
import com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.skyframe.AspectValue.AspectValueKey;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.FileDirtinessChecker;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.FileFunction.NonexistentFileReceiver;
import com.google.devtools.build.lib.skyframe.PackageFunction.ActionOnIOExceptionReadingBuildFile;
import com.google.devtools.build.lib.skyframe.PackageFunction.IncrementalityIntent;
import com.google.devtools.build.lib.skyframe.PackageFunction.LoadedPackageCacheEntry;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ActionCompletedReceiver;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ProgressSupplier;
import com.google.devtools.build.lib.skyframe.trimming.TrimmedConfigurationCache;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.ResourceUsage;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.BuildDriver;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.Differencer;
import com.google.devtools.build.skyframe.Differencer.DiffWithDelta.Delta;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.EventFilter;
import com.google.devtools.build.skyframe.GraphInconsistencyReceiver;
import com.google.devtools.build.skyframe.ImmutableDiff;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.Injectable;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator.EvaluatorSupplier;
import com.google.devtools.build.skyframe.NodeEntry;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.devtools.build.skyframe.WalkableGraph.WalkableGraphFactory;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;
import com.google.errorprone.annotations.ForOverride;
import java.io.IOException;
import java.io.PrintStream;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/**
 * A helper object to support Skyframe-driven execution.
 *
 * <p>This object is mostly used to inject external state, such as the executor engine or some
 * additional artifacts (workspace status and build info artifacts) into SkyFunctions for use during
 * the build.
 */
public abstract class SkyframeExecutor implements WalkableGraphFactory {
  private static final Logger logger = Logger.getLogger(SkyframeExecutor.class.getName());

  // We delete any value that can hold an action -- all subclasses of ActionLookupKey.
  protected static final Predicate<SkyKey> ANALYSIS_KEY_PREDICATE =
      k -> k instanceof ActionLookupValue.ActionLookupKey;

  private final EvaluatorSupplier evaluatorSupplier;
  protected MemoizingEvaluator memoizingEvaluator;
  private final MemoizingEvaluator.EmittedEventState emittedEventState =
      new MemoizingEvaluator.EmittedEventState();
  private final PackageFactory pkgFactory;
  private final WorkspaceStatusAction.Factory workspaceStatusActionFactory;
  private final FileSystem fileSystem;
  protected final BlazeDirectories directories;
  protected final ExternalFilesHelper externalFilesHelper;
  private final GraphInconsistencyReceiver graphInconsistencyReceiver;
  @Nullable protected OutputService outputService;

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
  // package twice (first time loading to find imported bzl files and declare Skyframe
  // dependencies).
  // TODO(bazel-team): remove this cache once we have skyframe-native package loading
  // [skyframe-loading]
  private final Cache<PackageIdentifier, LoadedPackageCacheEntry> packageFunctionCache =
      newPkgFunctionCache();
  private final Cache<PackageIdentifier, AstParseResult> astCache = newAstCache();

  private final AtomicInteger numPackagesLoaded = new AtomicInteger(0);
  @Nullable private final PackageProgressReceiver packageProgress;
  @Nullable private final ConfiguredTargetProgressReceiver configuredTargetProgress;

  private final SkyframeBuildView skyframeBuildView;
  private ActionLogBufferPathGenerator actionLogBufferPathGenerator;

  private BuildDriver buildDriver;

  private final Consumer<SkyframeExecutor> skyframeExecutorConsumerOnInit;

  // AtomicReferences are used here as mutable boxes shared with value builders.
  private final AtomicBoolean showLoadingProgress = new AtomicBoolean();
  protected final AtomicReference<UnixGlob.FilesystemCalls> syscalls =
      new AtomicReference<>(UnixGlob.DEFAULT_SYSCALLS);
  protected final AtomicReference<PathPackageLocator> pkgLocator = new AtomicReference<>();
  protected final AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages =
      new AtomicReference<>(ImmutableSet.<PackageIdentifier>of());
  private final AtomicReference<EventBus> eventBus = new AtomicReference<>();
  protected final AtomicReference<TimestampGranularityMonitor> tsgm = new AtomicReference<>();
  protected final AtomicReference<Map<String, String>> clientEnv = new AtomicReference<>();

  // Under normal circumstances, the artifact factory persists for the life of a Blaze server, but
  // since it is not yet created when we create the value builders, we have to use a supplier,
  // initialized when the build view is created.
  private final MutableArtifactFactorySupplier artifactFactory;
  private final ActionKeyContext actionKeyContext;

  protected boolean active = true;
  private final SkyframePackageManager packageManager;

  /** Used to lock evaluator on legacy calls to get existing values. */
  private final Object valueLookupLock = new Object();

  private final AtomicReference<ActionExecutionStatusReporter> statusReporterRef =
      new AtomicReference<>();
  private final SkyframeActionExecutor skyframeActionExecutor;
  private ActionExecutionFunction actionExecutionFunction;
  protected SkyframeProgressReceiver progressReceiver;
  private final AtomicReference<CyclesReporter> cyclesReporter = new AtomicReference<>();

  @VisibleForTesting boolean lastAnalysisDiscarded = false;

  private boolean analysisCacheDiscarded = false;

  private final ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions;

  protected SkyframeIncrementalBuildMonitor incrementalBuildMonitor =
      new SkyframeIncrementalBuildMonitor();

  private final ImmutableSet<PathFragment> hardcodedBlacklistedPackagePrefixes;
  private final PathFragment additionalBlacklistedPackagePrefixesFile;

  private final ConfiguredRuleClassProvider ruleClassProvider;

  private final CrossRepositoryLabelViolationStrategy crossRepositoryLabelViolationStrategy;

  private final List<BuildFileName> buildFilesByPriority;

  private final ActionOnIOExceptionReadingBuildFile actionOnIOExceptionReadingBuildFile;

  private final boolean shouldUnblockCpuWorkWhenFetchingDeps;

  private final BuildOptions defaultBuildOptions;

  private PerBuildSyscallCache perBuildSyscallCache;
  private int lastConcurrencyLevel = -1;

  private final PathResolverFactory pathResolverFactory = new PathResolverFactoryImpl();
  @Nullable private final NonexistentFileReceiver nonexistentFileReceiver;
  private final MutableSupplier<BigInteger> nonceVersion = new MutableSupplier<>();

  private final TrimmedConfigurationCache<SkyKey, Label, OptionsDiffForReconstruction>
      trimmingCache = TrimmedConfigurationProgressReceiver.buildCache();
  private final TrimmedConfigurationProgressReceiver trimmingListener =
      new TrimmedConfigurationProgressReceiver(trimmingCache);

  /** An {@link ArtifactResolverSupplier} that supports setting of an {@link ArtifactFactory}. */
  public static class MutableArtifactFactorySupplier implements ArtifactResolverSupplier {

    private ArtifactFactory artifactFactory;

    void set(ArtifactFactory artifactFactory) {
      this.artifactFactory = artifactFactory;
    }

    @Override
    public ArtifactFactory get() {
      return artifactFactory;
    }
  }

  class PathResolverFactoryImpl implements PathResolverFactory {
    @Override
    public boolean shouldCreatePathResolverForArtifactValues() {
      return outputService != null && outputService.supportsPathResolverForArtifactValues();
    }

    @Override
    public ArtifactPathResolver createPathResolverForArtifactValues(
        ActionInputMap actionInputMap,
        Map<Artifact, Collection<Artifact>> expandedArtifacts,
        Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesets,
        String workspaceName)
        throws IOException {
      Preconditions.checkState(shouldCreatePathResolverForArtifactValues());
      return outputService.createPathResolverForArtifactValues(
          directories.getExecRoot(workspaceName).asFragment(),
          directories.getRelativeOutputPath(),
          fileSystem,
          getPathEntries(),
          actionInputMap,
          expandedArtifacts,
          filesets);
    }
  }

  protected SkyframeExecutor(
      Consumer<SkyframeExecutor> skyframeExecutorConsumerOnInit,
      EvaluatorSupplier evaluatorSupplier,
      PackageFactory pkgFactory,
      FileSystem fileSystem,
      BlazeDirectories directories,
      ActionKeyContext actionKeyContext,
      Factory workspaceStatusActionFactory,
      ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions,
      ExternalFileAction externalFileAction,
      ImmutableSet<PathFragment> hardcodedBlacklistedPackagePrefixes,
      PathFragment additionalBlacklistedPackagePrefixesFile,
      CrossRepositoryLabelViolationStrategy crossRepositoryLabelViolationStrategy,
      List<BuildFileName> buildFilesByPriority,
      ActionOnIOExceptionReadingBuildFile actionOnIOExceptionReadingBuildFile,
      boolean shouldUnblockCpuWorkWhenFetchingDeps,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      BuildOptions defaultBuildOptions,
      @Nullable PackageProgressReceiver packageProgress,
      MutableArtifactFactorySupplier artifactResolverSupplier,
      @Nullable ConfiguredTargetProgressReceiver configuredTargetProgress,
      @Nullable NonexistentFileReceiver nonexistentFileReceiver,
      @Nullable ManagedDirectoriesKnowledge managedDirectoriesKnowledge) {
    // Strictly speaking, these arguments are not required for initialization, but all current
    // callsites have them at hand, so we might as well set them during construction.
    this.skyframeExecutorConsumerOnInit = skyframeExecutorConsumerOnInit;
    this.evaluatorSupplier = evaluatorSupplier;
    this.pkgFactory = pkgFactory;
    this.shouldUnblockCpuWorkWhenFetchingDeps = shouldUnblockCpuWorkWhenFetchingDeps;
    this.graphInconsistencyReceiver = graphInconsistencyReceiver;
    this.nonexistentFileReceiver = nonexistentFileReceiver;
    this.pkgFactory.setSyscalls(syscalls);
    this.workspaceStatusActionFactory = workspaceStatusActionFactory;
    this.packageManager =
        new SkyframePackageManager(
            new SkyframePackageLoader(),
            new SkyframeTransitivePackageLoader(),
            syscalls,
            cyclesReporter,
            pkgLocator,
            numPackagesLoaded,
            this);
    this.fileSystem = fileSystem;
    this.directories = Preconditions.checkNotNull(directories);
    this.actionKeyContext = Preconditions.checkNotNull(actionKeyContext);
    this.extraSkyFunctions = extraSkyFunctions;
    this.hardcodedBlacklistedPackagePrefixes = hardcodedBlacklistedPackagePrefixes;
    this.additionalBlacklistedPackagePrefixesFile = additionalBlacklistedPackagePrefixesFile;

    this.ruleClassProvider = (ConfiguredRuleClassProvider) pkgFactory.getRuleClassProvider();
    this.defaultBuildOptions = defaultBuildOptions;
    this.skyframeActionExecutor =
        new SkyframeActionExecutor(
            actionKeyContext, statusReporterRef, this::getPathEntries, this::createSourceArtifact);
    this.skyframeBuildView =
        new SkyframeBuildView(
            directories,
            this,
            (ConfiguredRuleClassProvider) ruleClassProvider,
            skyframeActionExecutor);
    this.artifactFactory = artifactResolverSupplier;
    this.artifactFactory.set(skyframeBuildView.getArtifactFactory());
    this.externalFilesHelper =
        ExternalFilesHelper.create(
            pkgLocator,
            externalFileAction,
            directories,
            managedDirectoriesKnowledge != null
                ? managedDirectoriesKnowledge
                : ManagedDirectoriesKnowledge.NO_MANAGED_DIRECTORIES);
    this.crossRepositoryLabelViolationStrategy = crossRepositoryLabelViolationStrategy;
    this.buildFilesByPriority = buildFilesByPriority;
    this.actionOnIOExceptionReadingBuildFile = actionOnIOExceptionReadingBuildFile;
    this.packageProgress = packageProgress;
    this.configuredTargetProgress = configuredTargetProgress;
  }

  private ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions(PackageFactory pkgFactory) {
    ConfiguredRuleClassProvider ruleClassProvider =
        (ConfiguredRuleClassProvider) pkgFactory.getRuleClassProvider();
    SkylarkImportLookupFunction skylarkImportLookupFunctionForInlining =
        getSkylarkImportLookupFunctionForInlining();
    // TODO(janakr): use this semaphore to bound memory usage for SkyFunctions besides
    // ConfiguredTargetFunction that may have a large temporary memory blow-up.
    Semaphore cpuBoundSemaphore = new Semaphore(ResourceUsage.getAvailableProcessors());

    // We don't check for duplicates in order to allow extraSkyfunctions to override existing
    // entries.
    Map<SkyFunctionName, SkyFunction> map = new HashMap<>();
    map.put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction());
    map.put(SkyFunctions.CLIENT_ENVIRONMENT_VARIABLE, new ClientEnvironmentFunction(clientEnv));
    map.put(SkyFunctions.ACTION_ENVIRONMENT_VARIABLE, new ActionEnvironmentFunction());
    map.put(FileStateValue.FILE_STATE, newFileStateFunction());
    map.put(SkyFunctions.DIRECTORY_LISTING_STATE, newDirectoryListingStateFunction());
    map.put(SkyFunctions.FILE_SYMLINK_CYCLE_UNIQUENESS, new FileSymlinkCycleUniquenessFunction());
    map.put(
        SkyFunctions.FILE_SYMLINK_INFINITE_EXPANSION_UNIQUENESS,
        new FileSymlinkInfiniteExpansionUniquenessFunction());
    map.put(FileValue.FILE, new FileFunction(pkgLocator, nonexistentFileReceiver));
    map.put(SkyFunctions.DIRECTORY_LISTING, new DirectoryListingFunction());
    map.put(
        SkyFunctions.PACKAGE_LOOKUP,
        new PackageLookupFunction(
            deletedPackages, crossRepositoryLabelViolationStrategy, buildFilesByPriority));
    map.put(SkyFunctions.CONTAINING_PACKAGE_LOOKUP, new ContainingPackageLookupFunction());
    map.put(SkyFunctions.AST_FILE_LOOKUP, new ASTFileLookupFunction(ruleClassProvider));
    map.put(
        SkyFunctions.SKYLARK_IMPORTS_LOOKUP,
        newSkylarkImportLookupFunction(ruleClassProvider, pkgFactory));
    map.put(SkyFunctions.GLOB, newGlobFunction());
    map.put(SkyFunctions.TARGET_PATTERN, new TargetPatternFunction());
    map.put(SkyFunctions.PREPARE_DEPS_OF_PATTERNS, new PrepareDepsOfPatternsFunction());
    map.put(
        SkyFunctions.PREPARE_DEPS_OF_PATTERN,
        new PrepareDepsOfPatternFunction(pkgLocator, traverseTestSuites()));
    map.put(
        SkyFunctions.PREPARE_TEST_SUITES_UNDER_DIRECTORY,
        new PrepareTestSuitesUnderDirectoryFunction(directories));
    map.put(
        SkyFunctions.PREPARE_DEPS_OF_TARGETS_UNDER_DIRECTORY,
        new PrepareDepsOfTargetsUnderDirectoryFunction(directories));
    map.put(SkyFunctions.COLLECT_TARGETS_IN_PACKAGE, new CollectTargetsInPackageFunction());
    map.put(SkyFunctions.COLLECT_TEST_SUITES_IN_PACKAGE, new CollectTestSuitesInPackageFunction());
    map.put(
        SkyFunctions.COLLECT_PACKAGES_UNDER_DIRECTORY,
        newCollectPackagesUnderDirectoryFunction(directories));
    map.put(
        SkyFunctions.BLACKLISTED_PACKAGE_PREFIXES,
        new BlacklistedPackagePrefixesFunction(
            hardcodedBlacklistedPackagePrefixes, additionalBlacklistedPackagePrefixesFile));
    map.put(SkyFunctions.TESTS_IN_SUITE, new TestExpansionFunction());
    map.put(SkyFunctions.TEST_SUITE_EXPANSION, new TestsForTargetPatternFunction());
    map.put(SkyFunctions.TARGET_PATTERN_PHASE, new TargetPatternPhaseFunction());
    map.put(
        SkyFunctions.PREPARE_ANALYSIS_PHASE,
        new PrepareAnalysisPhaseFunction(ruleClassProvider, defaultBuildOptions));
    map.put(SkyFunctions.RECURSIVE_PKG, new RecursivePkgFunction(directories));
    map.put(
        SkyFunctions.PACKAGE,
        new PackageFunction(
            pkgFactory,
            packageManager,
            showLoadingProgress,
            packageFunctionCache,
            astCache,
            numPackagesLoaded,
            skylarkImportLookupFunctionForInlining,
            packageProgress,
            actionOnIOExceptionReadingBuildFile,
            tracksStateForIncrementality()
                ? IncrementalityIntent.INCREMENTAL
                : IncrementalityIntent.NON_INCREMENTAL));
    map.put(SkyFunctions.PACKAGE_ERROR, new PackageErrorFunction());
    map.put(SkyFunctions.PACKAGE_ERROR_MESSAGE, new PackageErrorMessageFunction());
    map.put(SkyFunctions.TARGET_PATTERN_ERROR, new TargetPatternErrorFunction());
    map.put(SkyFunctions.TRANSITIVE_TARGET, new TransitiveTargetFunction(ruleClassProvider));
    map.put(Label.TRANSITIVE_TRAVERSAL, getTransitiveTraversalFunction());
    map.put(
        SkyFunctions.CONFIGURED_TARGET,
        new ConfiguredTargetFunction(
            new BuildViewProvider(),
            ruleClassProvider,
            cpuBoundSemaphore,
            shouldStoreTransitivePackagesInLoadingAndAnalysis(),
            shouldUnblockCpuWorkWhenFetchingDeps,
            defaultBuildOptions,
            configuredTargetProgress,
            nonceVersion));
    map.put(
        SkyFunctions.ASPECT,
        new AspectFunction(
            new BuildViewProvider(),
            ruleClassProvider,
            skylarkImportLookupFunctionForInlining,
            shouldStoreTransitivePackagesInLoadingAndAnalysis(),
            defaultBuildOptions,
            nonceVersion));
    map.put(
        SkyFunctions.LOAD_SKYLARK_ASPECT,
        new ToplevelSkylarkAspectFunction(skylarkImportLookupFunctionForInlining));
    map.put(
        SkyFunctions.POST_CONFIGURED_TARGET,
        new PostConfiguredTargetFunction(
            new BuildViewProvider(), ruleClassProvider, defaultBuildOptions));
    map.put(
        SkyFunctions.BUILD_CONFIGURATION,
        new BuildConfigurationFunction(directories, ruleClassProvider, defaultBuildOptions));
    map.put(SkyFunctions.WORKSPACE_NAME, new WorkspaceNameFunction());
    map.put(SkyFunctions.WORKSPACE_AST, new WorkspaceASTFunction(ruleClassProvider));
    map.put(
        WorkspaceFileValue.WORKSPACE_FILE,
        new WorkspaceFileFunction(
            ruleClassProvider, pkgFactory, directories, skylarkImportLookupFunctionForInlining));
    map.put(SkyFunctions.EXTERNAL_PACKAGE, new ExternalPackageFunction());
    map.put(
        SkyFunctions.TARGET_COMPLETION,
        CompletionFunction.targetCompletionFunction(
            pathResolverFactory, skyframeActionExecutor::getExecRoot));
    map.put(
        SkyFunctions.ASPECT_COMPLETION,
        CompletionFunction.aspectCompletionFunction(
            pathResolverFactory, skyframeActionExecutor::getExecRoot));
    map.put(SkyFunctions.TEST_COMPLETION, new TestCompletionFunction());
    map.put(
        Artifact.ARTIFACT,
        new ArtifactFunction(
            () -> !skyframeActionExecutor.actionFileSystemType().inMemoryFileSystem()));
    map.put(
        SkyFunctions.BUILD_INFO_COLLECTION,
        new BuildInfoCollectionFunction(actionKeyContext, artifactFactory::get));
    map.put(SkyFunctions.BUILD_INFO, new WorkspaceStatusFunction(this::makeWorkspaceStatusAction));
    map.put(SkyFunctions.COVERAGE_REPORT, new CoverageReportFunction(actionKeyContext));
    ActionExecutionFunction actionExecutionFunction =
        new ActionExecutionFunction(skyframeActionExecutor, directories, tsgm);
    map.put(SkyFunctions.ACTION_EXECUTION, actionExecutionFunction);
    this.actionExecutionFunction = actionExecutionFunction;
    map.put(SkyFunctions.ACTION_SKETCH, new ActionSketchFunction(actionKeyContext));
    map.put(
        SkyFunctions.RECURSIVE_FILESYSTEM_TRAVERSAL, new RecursiveFilesystemTraversalFunction());
    map.put(SkyFunctions.FILESET_ENTRY, new FilesetEntryFunction(directories::getExecRoot));
    map.put(
        SkyFunctions.ACTION_TEMPLATE_EXPANSION,
        new ActionTemplateExpansionFunction(actionKeyContext));
    map.put(SkyFunctions.LOCAL_REPOSITORY_LOOKUP, new LocalRepositoryLookupFunction());
    map.put(
        SkyFunctions.REGISTERED_EXECUTION_PLATFORMS, new RegisteredExecutionPlatformsFunction());
    map.put(SkyFunctions.REGISTERED_TOOLCHAINS, new RegisteredToolchainsFunction());
    map.put(SkyFunctions.SINGLE_TOOLCHAIN_RESOLUTION, new SingleToolchainResolutionFunction());
    map.put(SkyFunctions.TOOLCHAIN_RESOLUTION, new ToolchainResolutionFunction());
    map.put(SkyFunctions.REPOSITORY_MAPPING, new RepositoryMappingFunction());
    map.put(SkyFunctions.RESOLVED_HASH_VALUES, new ResolvedHashesFunction());
    map.put(SkyFunctions.RESOLVED_FILE, new ResolvedFileFunction());
    map.put(SkyFunctions.PLATFORM_MAPPING, new PlatformMappingFunction());
    map.put(SkyFunctions.ARTIFACT_NESTED_SET, ArtifactNestedSetFunction.createInstance());
    map.putAll(extraSkyFunctions);
    return ImmutableMap.copyOf(map);
  }

  protected SkyFunction getTransitiveTraversalFunction() {
    return new TransitiveTraversalFunction();
  }

  protected boolean traverseTestSuites() {
    return false;
  }

  protected SkyFunction newFileStateFunction() {
    return new FileStateFunction(tsgm, syscalls, externalFilesHelper);
  }

  protected SkyFunction newDirectoryListingStateFunction() {
    return new DirectoryListingStateFunction(externalFilesHelper, syscalls);
  }

  protected SkyFunction newGlobFunction() {
    return new GlobFunction(/*alwaysUseDirListing=*/ false);
  }

  protected SkyFunction newCollectPackagesUnderDirectoryFunction(BlazeDirectories directories) {
    return new CollectPackagesUnderDirectoryFunction(directories);
  }

  @Nullable
  protected SkylarkImportLookupFunction getSkylarkImportLookupFunctionForInlining() {
    return null;
  }

  protected SkyFunction newSkylarkImportLookupFunction(
      RuleClassProvider ruleClassProvider, PackageFactory pkgFactory) {
    return new SkylarkImportLookupFunction(ruleClassProvider, this.pkgFactory);
  }

  protected PerBuildSyscallCache newPerBuildSyscallCache(int concurrencyLevel) {
    return PerBuildSyscallCache.newBuilder().setConcurrencyLevel(concurrencyLevel).build();
  }

  /**
   * Gets a (possibly cached) syscalls cache, re-initialized each build.
   *
   * <p>We cache the syscalls cache if possible because construction of the cache is surprisingly
   * expensive, and is on the critical path of null builds.
   */
  protected final PerBuildSyscallCache getPerBuildSyscallCache(int concurrencyLevel) {
    if (perBuildSyscallCache != null && lastConcurrencyLevel == concurrencyLevel) {
      perBuildSyscallCache.clear();
      return perBuildSyscallCache;
    }
    lastConcurrencyLevel = concurrencyLevel;
    perBuildSyscallCache = newPerBuildSyscallCache(concurrencyLevel);
    return perBuildSyscallCache;
  }

  /** Do not use except in subclasses. */
  protected void setNonceVersion(BigInteger nonceVersion) {
    this.nonceVersion.set(nonceVersion);
  }

  @ThreadCompatible
  public void setActive(boolean active) {
    this.active = active;
  }

  protected void checkActive() {
    Preconditions.checkState(active);
  }

  public void configureActionExecutor(
      MetadataProvider fileCache, ActionInputPrefetcher actionInputPrefetcher) {
    this.skyframeActionExecutor.configure(fileCache, actionInputPrefetcher);
  }

  public void dump(boolean summarize, PrintStream out) {
    memoizingEvaluator.dump(summarize, out);
  }

  public abstract void dumpPackages(PrintStream out);

  public void setOutputService(OutputService outputService) {
    this.outputService = outputService;
  }

  /** Inform this SkyframeExecutor that a new command is starting. */
  public void noteCommandStart() {}

  /** Notify listeners about changed files, and release any associated memory afterwards. */
  public void drainChangedFiles() {
    incrementalBuildMonitor.alertListeners(getEventBus());
    incrementalBuildMonitor = null;
  }

  public final BuildDriver getDriver() {
    return buildDriver;
  }

  public boolean wasAnalysisCacheDiscardedAndResetBit() {
    boolean tmp = analysisCacheDiscarded;
    analysisCacheDiscarded = false;
    return tmp;
  }

  /**
   * This method exists only to allow a module to make a top-level Skyframe call during the
   * transition to making it fully Skyframe-compatible. Do not add additional callers!
   */
  public SkyValue evaluateSkyKeyForExecutionSetup(
      final ExtendedEventHandler eventHandler, final SkyKey key)
      throws EnvironmentalExecException, InterruptedException {
    synchronized (valueLookupLock) {
      // We evaluate in keepGoing mode because in the case that the graph does not store its
      // edges, nokeepGoing builds are not allowed, whereas keepGoing builds are always
      // permitted.
      EvaluationResult<?> result =
          evaluate(
              ImmutableList.of(key), true, ResourceUsage.getAvailableProcessors(), eventHandler);
      if (!result.hasError()) {
        return Preconditions.checkNotNull(result.get(key), "%s %s", result, key);
      }
      ErrorInfo errorInfo = Preconditions.checkNotNull(result.getError(key), "%s %s", key, result);
      Throwables.propagateIfPossible(errorInfo.getException(), EnvironmentalExecException.class);
      if (errorInfo.getException() != null) {
        throw new IllegalStateException(errorInfo.getException());
      }
      throw new IllegalStateException(errorInfo.toString());
    }
  }

  public abstract ActionGraphContainer getActionGraphContainer(
      List<String> actionGraphTargets, boolean includeActionCmdLine, boolean includeArtifacts)
      throws CommandLineExpansionException;

  class BuildViewProvider {
    /** Returns the current {@link SkyframeBuildView} instance. */
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
    ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions = skyFunctions(pkgFactory);
    memoizingEvaluator =
        evaluatorSupplier.create(
            skyFunctions,
            evaluatorDiffer(),
            progressReceiver,
            graphInconsistencyReceiver,
            DEFAULT_FILTER_WITH_ACTIONS,
            emittedEventState,
            tracksStateForIncrementality());
    buildDriver = createBuildDriver();
    skyframeExecutorConsumerOnInit.accept(this);
  }

  /**
   * Use the fact that analysis of a target must occur before execution of that target, and in a
   * separate Skyframe evaluation, to avoid propagating events from configured target nodes (and
   * more generally action lookup nodes) to action execution nodes. We take advantage of the fact
   * that if a node depends on an action lookup node and is not itself an action lookup node, then
   * it is an execution-phase node: the action lookup nodes are terminal in the analysis phase.
   */
  private static final EventFilter DEFAULT_FILTER_WITH_ACTIONS =
      new EventFilter() {
        @Override
        public boolean storeEventsAndPosts() {
          return true;
        }

        @Override
        public boolean apply(Event input) {
          // Use the filtering defined in the default filter: no info/progress messages.
          return InMemoryMemoizingEvaluator.DEFAULT_STORED_EVENT_FILTER.apply(input);
        }

        @Override
        public Predicate<SkyKey> depEdgeFilterForEventsAndPosts(SkyKey primaryKey) {
          if (primaryKey instanceof ActionLookupValue.ActionLookupKey) {
            return Predicates.alwaysTrue();
          } else {
            return NO_ACTION_LOOKUP;
          }
        }
      };

  private static final Predicate<SkyKey> NO_ACTION_LOOKUP =
      (key) -> !(key instanceof ActionLookupValue.ActionLookupKey);

  protected SkyframeProgressReceiver newSkyframeProgressReceiver() {
    return new SkyframeProgressReceiver();
  }

  /** Reinitializes the Skyframe evaluator, dropping all previously computed values. */
  public void resetEvaluator() {
    init();
    emittedEventState.clear();
    clearTrimmingCache();
    skyframeBuildView.reset();
  }

  /**
   * Notifies the executor that the command is complete. May safely be called multiple times for a
   * single command, so callers should err on the side of calling it more frequently. Should be
   * idempotent, so that calls after the first one in the same evaluation should be quick.
   */
  public void notifyCommandComplete(ExtendedEventHandler eventHandler) throws InterruptedException {
    memoizingEvaluator.noteEvaluationsAtSameVersionMayBeFinished(eventHandler);
  }

  /**
   * Notifies the executor to post logging stats when the server is crashing, so that logging is
   * still available even when the server crashes.
   */
  public void postLoggingStatsWhenCrashing(ExtendedEventHandler eventHandler) {
    memoizingEvaluator.postLoggingStats(eventHandler);
  }

  protected abstract Differencer evaluatorDiffer();

  @ForOverride
  protected abstract BuildDriver createBuildDriver();

  /** Clear any configured target data stored outside Skyframe. */
  public void handleAnalysisInvalidatingChange() {
    logger.info("Dropping configured target data");
    analysisCacheDiscarded = true;
    clearTrimmingCache();
    skyframeBuildView.clearInvalidatedConfiguredTargets();
    skyframeBuildView.clearLegacyData();
  }

  /** Activates retroactive trimming (idempotently, so has no effect if already active). */
  void activateRetroactiveTrimming() {
    trimmingListener.activate();
  }

  /** Deactivates retroactive trimming (idempotently, so has no effect if already inactive). */
  void deactivateRetroactiveTrimming() {
    trimmingListener.deactivate();
  }

  protected void clearTrimmingCache() {
    trimmingCache.clear();
  }

  /** Used with dump --rules. */
  public static class RuleStat {
    private final String key;
    private final String name;
    private final boolean isRule;
    private long count;
    private long actionCount;

    public RuleStat(String key, String name, boolean isRule) {
      this.key = key;
      this.name = name;
      this.isRule = isRule;
    }

    public void addRule(long actionCount) {
      this.count++;
      this.actionCount += actionCount;
    }

    /** Returns a key that uniquely identifies this rule or aspect. */
    public String getKey() {
      return key;
    }

    /** Returns a name for the rule or aspect. */
    public String getName() {
      return name;
    }

    /** Returns whether this is a rule or an aspect. */
    public boolean isRule() {
      return isRule;
    }

    /** Returns the instance count of this rule or aspect class. */
    public long getCount() {
      return count;
    }

    /** Returns the total action count of all instance of this rule or aspect class. */
    public long getActionCount() {
      return actionCount;
    }
  }

  /** Computes statistics on heap-resident rules and aspects. */
  public abstract List<RuleStat> getRuleStats(ExtendedEventHandler eventHandler);

  /**
   * Decides if graph edges should be stored during this evaluation and checks if the state from the
   * last evaluation, if any, can be kept.
   *
   * <p>If not, it will mark this state for deletion. The actual cleaning is put off until {@link
   * #sync}, in case no evaluation was actually called for and the existing state can be kept for
   * longer.
   */
  public void decideKeepIncrementalState(
      boolean batch,
      boolean keepStateAfterBuild,
      boolean trackIncrementalState,
      boolean discardAnalysisCache,
      EventHandler eventHandler) {
    // Assume incrementality.
  }

  /** Whether this executor tracks state for the purpose of improving incremental performance. */
  public boolean tracksStateForIncrementality() {
    return true;
  }

  /**
   * If not null, this is the only source root in the build, corresponding to the single element in
   * a single-element package path. Such a single-source-root build need not plant the execroot
   * symlink forest, and can trivially resolve source artifacts from exec paths. As a consequence,
   * builds where this is not null do not need to track a package -> source root map, and so do not
   * need to track all loaded packages.
   */
  @Nullable
  protected Root getForcedSingleSourceRootIfNoExecrootSymlinkCreation() {
    return null;
  }

  private boolean shouldStoreTransitivePackagesInLoadingAndAnalysis() {
    return getForcedSingleSourceRootIfNoExecrootSymlinkCreation() == null;
  }

  @VisibleForTesting
  protected abstract Injectable injectable();

  /**
   * Types that are created during loading, use significant space, and are definitely not needed
   * during execution unless explicitly named.
   *
   * <p>Some keys, like globs, may be re-evaluated during execution, so these types should only be
   * discarded if reverse deps are not being tracked!
   */
  private static final ImmutableSet<SkyFunctionName> LOADING_TYPES =
      ImmutableSet.of(
          SkyFunctions.PACKAGE,
          SkyFunctions.SKYLARK_IMPORTS_LOOKUP,
          SkyFunctions.AST_FILE_LOOKUP,
          SkyFunctions.GLOB);

  /** Data that should be discarded in {@link #discardPreExecutionCache}. */
  protected enum DiscardType {
    ALL,
    ANALYSIS_REFS_ONLY,
    LOADING_NODES_ONLY;

    boolean discardsAnalysis() {
      return this != LOADING_NODES_ONLY;
    }

    boolean discardsLoading() {
      return this != ANALYSIS_REFS_ONLY;
    }
  }

  /**
   * Save memory by removing references to configured targets and aspects in Skyframe.
   *
   * <p>These nodes must be recreated on subsequent builds. We do not clear the top-level target
   * nodes, since their configured targets are needed for the target completion middleman values.
   *
   * <p>The nodes are not deleted during this method call, because they are needed for the execution
   * phase. Instead, their analysis-time data is cleared while preserving the generating action info
   * needed for execution. The next build will delete the nodes (and recreate them if necessary).
   *
   * <p>{@code discardType} can be used to specify which data to discard.
   */
  protected void discardPreExecutionCache(
      Collection<ConfiguredTarget> topLevelTargets,
      Collection<AspectValue> topLevelAspects,
      DiscardType discardType) {
    if (discardType.discardsAnalysis()) {
      topLevelTargets = ImmutableSet.copyOf(topLevelTargets);
      topLevelAspects = ImmutableSet.copyOf(topLevelAspects);
    }
    // This is to prevent throwing away Packages we may need during execution.
    ImmutableSet.Builder<PackageIdentifier> packageSetBuilder = ImmutableSet.builder();
    if (discardType.discardsLoading()) {
      packageSetBuilder.addAll(
          Collections2.transform(
              topLevelTargets, target -> target.getLabel().getPackageIdentifier()));
      packageSetBuilder.addAll(
          Collections2.transform(
              topLevelAspects, aspect -> aspect.getLabel().getPackageIdentifier()));
    }
    ImmutableSet<PackageIdentifier> topLevelPackages = packageSetBuilder.build();
    try (AutoProfiler p = AutoProfiler.logged("discarding analysis cache", logger)) {
      lastAnalysisDiscarded = true;
      Iterator<? extends Map.Entry<SkyKey, ? extends NodeEntry>> it =
          memoizingEvaluator.getGraphEntries().iterator();
      while (it.hasNext()) {
        Map.Entry<SkyKey, ? extends NodeEntry> keyAndEntry = it.next();
        NodeEntry entry = keyAndEntry.getValue();
        if (entry == null || !entry.isDone()) {
          continue;
        }
        SkyKey key = keyAndEntry.getKey();
        SkyFunctionName functionName = key.functionName();
        if (discardType.discardsLoading()) {
          // Keep packages for top-level targets and aspects in memory to get the target from later.
          if (functionName.equals(SkyFunctions.PACKAGE)
              && topLevelPackages.contains(key.argument())) {
            continue;
          }
          if (LOADING_TYPES.contains(functionName)) {
            it.remove();
            continue;
          }
        }
        if (discardType.discardsAnalysis()) {
          if (functionName.equals(SkyFunctions.CONFIGURED_TARGET)) {
            ConfiguredTargetValue ctValue;
            try {
              ctValue = (ConfiguredTargetValue) entry.getValue();
            } catch (InterruptedException e) {
              throw new IllegalStateException(
                  "No interruption in in-memory retrieval: " + entry, e);
            }
            // ctValue may be null if target was not successfully analyzed.
            if (ctValue != null) {
              ctValue.clear(!topLevelTargets.contains(ctValue.getConfiguredTarget()));
            }
          } else if (functionName.equals(SkyFunctions.ASPECT)) {
            AspectValue aspectValue;
            try {
              aspectValue = (AspectValue) entry.getValue();
            } catch (InterruptedException e) {
              throw new IllegalStateException(
                  "No interruption in in-memory retrieval: " + entry, e);
            }
            // value may be null if target was not successfully analyzed.
            if (aspectValue != null) {
              aspectValue.clear(!topLevelAspects.contains(aspectValue));
            }
          }
        }
      }
    }
  }

  /**
   * Saves memory by clearing analysis objects from Skyframe. Clears their data without deleting
   * them (they will be deleted on the next build). May also delete loading-phase objects from the
   * graph.
   */
  public abstract void clearAnalysisCache(
      Collection<ConfiguredTarget> topLevelTargets, Collection<AspectValue> topLevelAspects);

  protected abstract void dropConfiguredTargetsNow(final ExtendedEventHandler eventHandler);

  private WorkspaceStatusAction makeWorkspaceStatusAction(String workspaceName) {
    WorkspaceStatusAction.Environment env =
        new WorkspaceStatusAction.Environment() {
          @Override
          public Artifact createStableArtifact(String name) {
            ArtifactRoot root = directories.getBuildDataDirectory(workspaceName);
            return skyframeBuildView
                .getArtifactFactory()
                .getDerivedArtifact(
                    PathFragment.create(name), root, WorkspaceStatusValue.BUILD_INFO_KEY);
          }

          @Override
          public Artifact createVolatileArtifact(String name) {
            ArtifactRoot root = directories.getBuildDataDirectory(workspaceName);
            return skyframeBuildView
                .getArtifactFactory()
                .getConstantMetadataArtifact(
                    PathFragment.create(name), root, WorkspaceStatusValue.BUILD_INFO_KEY);
          }
        };
    return workspaceStatusActionFactory.createWorkspaceStatusAction(env);
  }

  @VisibleForTesting
  public ArtifactResolverSupplier getArtifactResolverSupplierForTesting() {
    return artifactFactory;
  }

  @VisibleForTesting
  @Nullable
  public WorkspaceStatusAction getLastWorkspaceStatusAction() throws InterruptedException {
    WorkspaceStatusValue workspaceStatusValue =
        (WorkspaceStatusValue)
            memoizingEvaluator.getExistingValue(WorkspaceStatusValue.BUILD_INFO_KEY);
    return workspaceStatusValue == null
        ? null
        : (WorkspaceStatusAction) workspaceStatusValue.getAction(0);
  }

  public void injectCoverageReportData(Actions.GeneratingActions actions) {
    PrecomputedValue.COVERAGE_REPORT_KEY.set(injectable(), actions.getActions());
  }

  private void setDefaultVisibility(RuleVisibility defaultVisibility) {
    PrecomputedValue.DEFAULT_VISIBILITY.set(injectable(), defaultVisibility);
  }

  private void setSkylarkSemantics(StarlarkSemantics starlarkSemantics) {
    PrecomputedValue.STARLARK_SEMANTICS.set(injectable(), starlarkSemantics);
  }

  public void injectExtraPrecomputedValues(List<PrecomputedValue.Injected> extraPrecomputedValues) {
    for (PrecomputedValue.Injected injected : extraPrecomputedValues) {
      injected.inject(injectable());
    }
  }

  protected Cache<PackageIdentifier, LoadedPackageCacheEntry> newPkgFunctionCache() {
    return CacheBuilder.newBuilder().build();
  }

  protected Cache<PackageIdentifier, AstParseResult> newAstCache() {
    return CacheBuilder.newBuilder().build();
  }

  private void setShowLoadingProgress(boolean showLoadingProgressValue) {
    showLoadingProgress.set(showLoadingProgressValue);
  }

  protected void setCommandId(UUID commandId) {
    PrecomputedValue.BUILD_ID.set(injectable(), commandId);
  }

  /** Returns the build-info.txt and build-changelist.txt artifacts. */
  public Collection<Artifact> getWorkspaceStatusArtifacts(ExtendedEventHandler eventHandler)
      throws InterruptedException {
    // Should already be present, unless the user didn't request any targets for analysis.
    EvaluationResult<WorkspaceStatusValue> result =
        evaluate(
            ImmutableList.of(WorkspaceStatusValue.BUILD_INFO_KEY),
            /*keepGoing=*/ true,
            /*numThreads=*/ 1,
            eventHandler);
    WorkspaceStatusValue value =
        Preconditions.checkNotNull(result.get(WorkspaceStatusValue.BUILD_INFO_KEY));
    return ImmutableList.of(value.getStableArtifact(), value.getVolatileArtifact());
  }

  public Map<PathFragment, Root> getArtifactRootsForFiles(
      final ExtendedEventHandler eventHandler, Iterable<PathFragment> execPaths)
      throws InterruptedException {
    return getArtifactRoots(eventHandler, execPaths, true);
  }

  public Map<PathFragment, Root> getArtifactRoots(
      final ExtendedEventHandler eventHandler, Iterable<PathFragment> execPaths)
      throws InterruptedException {
    return getArtifactRoots(eventHandler, execPaths, false);
  }

  private Map<PathFragment, Root> getArtifactRoots(
      final ExtendedEventHandler eventHandler, Iterable<PathFragment> execPaths, boolean forFiles)
      throws InterruptedException {
    final Map<PathFragment, SkyKey> packageKeys = new HashMap<>();
    for (PathFragment execPath : execPaths) {
      try {
        PackageIdentifier pkgIdentifier =
            PackageIdentifier.discoverFromExecPath(execPath, forFiles);
        packageKeys.put(execPath, ContainingPackageLookupValue.key(pkgIdentifier));
      } catch (LabelSyntaxException e) {
        continue;
      }
    }

    EvaluationResult<ContainingPackageLookupValue> result;
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(true)
            .setNumThreads(1)
            .setEventHander(eventHandler)
            .build();
    synchronized (valueLookupLock) {
      result = buildDriver.evaluate(packageKeys.values(), evaluationContext);
    }

    if (result.hasError()) {
      return new HashMap<>();
    }

    Map<PathFragment, Root> roots = new HashMap<>();
    for (PathFragment execPath : execPaths) {
      ContainingPackageLookupValue value = result.get(packageKeys.get(execPath));
      if (value.hasContainingPackage()) {
        roots.put(
            execPath,
            maybeTransformRootForRepository(
                value.getContainingPackageRoot(),
                value.getContainingPackageName().getRepository()));
      } else {
        roots.put(execPath, null);
      }
    }
    return roots;
  }

  // This must always be consistent with Package.getSourceRoot; otherwise computing source roots
  // from exec paths does not work, which can break the action cache for input-discovering actions.
  static Root maybeTransformRootForRepository(Root packageRoot, RepositoryName repository) {
    if (repository.isMain()) {
      return packageRoot;
    } else {
      Path actualRootPath = packageRoot.asPath();
      int segmentCount = repository.getSourceRoot().segmentCount();
      for (int i = 0; i < segmentCount; i++) {
        actualRootPath = actualRootPath.getParentDirectory();
      }
      return Root.fromPath(actualRootPath);
    }
  }

  @VisibleForTesting
  public SkyFunctionEnvironmentForTesting getSkyFunctionEnvironmentForTesting(
      ExtendedEventHandler eventHandler) {
    return new SkyFunctionEnvironmentForTesting(eventHandler, this);
  }

  public EventBus getEventBus() {
    return eventBus.get();
  }

  @VisibleForTesting
  ImmutableList<Root> getPathEntries() {
    return pkgLocator.get().getPathEntries();
  }

  private SourceArtifact createSourceArtifact(PathFragment execPath) {
    // This is only used by ActionFileSystem.
    return artifactFactory
        .get()
        .getSourceArtifact(execPath, Iterables.getOnlyElement(getPathEntries()));
  }

  public AtomicReference<PathPackageLocator> getPackageLocator() {
    return pkgLocator;
  }

  protected abstract void invalidate(Predicate<SkyKey> pred);

  private static boolean compatibleFileTypes(Dirent.Type oldType, FileStateType newType) {
    return (oldType.equals(Dirent.Type.FILE) && newType.equals(FileStateType.REGULAR_FILE))
        || (oldType.equals(Dirent.Type.UNKNOWN) && newType.equals(FileStateType.SPECIAL_FILE))
        || (oldType.equals(Dirent.Type.DIRECTORY) && newType.equals(FileStateType.DIRECTORY))
        || (oldType.equals(Dirent.Type.SYMLINK) && newType.equals(FileStateType.SYMLINK));
  }

  protected Differencer.Diff getDiff(
      TimestampGranularityMonitor tsgm,
      Iterable<PathFragment> modifiedSourceFiles,
      final Root pathEntry)
      throws InterruptedException {
    if (Iterables.isEmpty(modifiedSourceFiles)) {
      return new ImmutableDiff(ImmutableList.<SkyKey>of(), ImmutableMap.<SkyKey, SkyValue>of());
    }
    // TODO(bazel-team): change ModifiedFileSet to work with RootedPaths instead of PathFragments.
    Iterable<SkyKey> dirtyFileStateSkyKeys =
        Iterables.transform(
            modifiedSourceFiles,
            new Function<PathFragment, SkyKey>() {
              @Override
              public SkyKey apply(PathFragment pathFragment) {
                Preconditions.checkState(
                    !pathFragment.isAbsolute(), "found absolute PathFragment: %s", pathFragment);
                return FileStateValue.key(RootedPath.toRootedPath(pathEntry, pathFragment));
              }
            });
    // We only need to invalidate directory values when a file has been created or deleted or
    // changes type, not when it has merely been modified. Unfortunately we do not have that
    // information here, so we compute it ourselves.
    // TODO(bazel-team): Fancy filesystems could provide it with a hypothetically modified
    // DiffAwareness interface.
    logger.info(
        "About to recompute filesystem nodes corresponding to files that are known to have "
            + "changed");
    FilesystemValueChecker fsvc = new FilesystemValueChecker(tsgm, null);
    Map<SkyKey, SkyValue> valuesMap = memoizingEvaluator.getValues();
    Differencer.DiffWithDelta diff =
        fsvc.getNewAndOldValues(valuesMap, dirtyFileStateSkyKeys, new FileDirtinessChecker());

    Set<SkyKey> valuesToInvalidate = new HashSet<>();
    Map<SkyKey, SkyValue> valuesToInject = new HashMap<>();
    for (Map.Entry<SkyKey, Delta> entry : diff.changedKeysWithNewAndOldValues().entrySet()) {
      SkyKey key = entry.getKey();
      Preconditions.checkState(key.functionName().equals(FileStateValue.FILE_STATE), key);
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
            (DirectoryListingStateValue) valuesMap.get(dirListingStateKey);
        if (oldDirListingStateValue != null) {
          String baseName = rootedPath.getRootRelativePath().getBaseName();
          Dirent oldDirent = oldDirListingStateValue.getDirents().maybeGetDirent(baseName);
          changedType =
              (oldDirent == null) || !compatibleFileTypes(oldDirent.getType(), newValue.getType());
        } else {
          changedType = true;
        }
      }
      if (changedType) {
        valuesToInvalidate.add(dirListingStateKey);
      }
    }
    for (SkyKey key : diff.changedKeysWithoutNewValues()) {
      Preconditions.checkState(key.functionName().equals(FileStateValue.FILE_STATE), key);
      RootedPath rootedPath = (RootedPath) key.argument();
      valuesToInvalidate.add(parentDirectoryListingStateKey(rootedPath));
    }
    return new ImmutableDiff(valuesToInvalidate, valuesToInject);
  }

  private static SkyKey parentDirectoryListingStateKey(RootedPath rootedPath) {
    RootedPath parentDirRootedPath = rootedPath.getParentDirectory();
    return DirectoryListingStateValue.key(parentDirRootedPath);
  }

  /** Sets the packages that should be treated as deleted and ignored. */
  @VisibleForTesting // productionVisibility = Visibility.PRIVATE
  public abstract void setDeletedPackages(Iterable<PackageIdentifier> pkgs);

  /**
   * Prepares the evaluator for loading.
   *
   * <p>MUST be run before every incremental build.
   */
  @VisibleForTesting // productionVisibility = Visibility.PRIVATE
  public void preparePackageLoading(
      PathPackageLocator pkgLocator,
      PackageCacheOptions packageCacheOptions,
      StarlarkSemanticsOptions starlarkSemanticsOptions,
      UUID commandId,
      Map<String, String> clientEnv,
      TimestampGranularityMonitor tsgm) {
    Preconditions.checkNotNull(pkgLocator);
    Preconditions.checkNotNull(tsgm);
    setActive(true);

    this.tsgm.set(tsgm);
    setCommandId(commandId);
    this.clientEnv.set(clientEnv);

    setShowLoadingProgress(packageCacheOptions.showLoadingProgress);
    setDefaultVisibility(packageCacheOptions.defaultVisibility);
    setSkylarkSemantics(getEffectiveStarlarkSemantics(starlarkSemanticsOptions));
    setPackageLocator(pkgLocator);

    syscalls.set(getPerBuildSyscallCache(packageCacheOptions.globbingThreads));
    this.pkgFactory.setGlobbingThreads(packageCacheOptions.globbingThreads);
    this.pkgFactory.setMaxDirectoriesToEagerlyVisitInGlobbing(
        packageCacheOptions.maxDirectoriesToEagerlyVisitInGlobbing);
    emittedEventState.clear();

    // If the PackageFunction was interrupted, there may be stale entries here.
    packageFunctionCache.invalidateAll();
    astCache.invalidateAll();
    numPackagesLoaded.set(0);
    if (packageProgress != null) {
      packageProgress.reset();
    }

    // Reset the stateful SkyframeCycleReporter, which contains cycles from last run.
    cyclesReporter.set(createCyclesReporter());
  }

  public StarlarkSemantics getEffectiveStarlarkSemantics(
      StarlarkSemanticsOptions starlarkSemanticsOptions) {
    return starlarkSemanticsOptions.toSkylarkSemantics();
  }

  private void setPackageLocator(PathPackageLocator pkgLocator) {
    EventBus eventBus = this.eventBus.get();
    if (eventBus != null) {
      eventBus.post(pkgLocator);
    }

    PathPackageLocator oldLocator = this.pkgLocator.getAndSet(pkgLocator);
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(injectable(), pkgLocator);

    if (!pkgLocator.equals(oldLocator)) {
      // The package path is read not only by SkyFunctions but also by some other code paths.
      // We need to take additional steps to keep the corresponding data structures in sync.
      // (Some of the additional steps are carried out by ConfiguredTargetValueInvalidationListener,
      // and some by BuildView#buildHasIncompatiblePackageRoots and #updateSkyframe.)
      artifactFactory
          .get()
          .setSourceArtifactRoots(
              createSourceArtifactRootMapOnNewPkgLocator(oldLocator, pkgLocator));
    }
  }

  protected ImmutableMap<Root, ArtifactRoot> createSourceArtifactRootMapOnNewPkgLocator(
      PathPackageLocator oldLocator, PathPackageLocator pkgLocator) {
    // TODO(bazel-team): The output base is a legitimate "source root" because external repositories
    // stage their sources under output_base/external. The root here should really be
    // output_base/external, but for some reason it isn't.
    return Stream.concat(
            pkgLocator.getPathEntries().stream(),
            Stream.of(Root.absoluteRoot(fileSystem), Root.fromPath(directories.getOutputBase())))
        .distinct()
        .collect(
            ImmutableMap.toImmutableMap(
                java.util.function.Function.identity(), ArtifactRoot::asSourceRoot));
  }

  public SkyframeBuildView getSkyframeBuildView() {
    return skyframeBuildView;
  }

  /** Sets the eventBus to use for posting events. */
  public void setEventBus(EventBus eventBus) {
    this.eventBus.set(eventBus);
  }

  public void setClientEnv(Map<String, String> clientEnv) {
    this.skyframeActionExecutor.setClientEnv(clientEnv);
  }

  /** Sets the path for action log buffers. */
  public void setActionOutputRoot(Path actionOutputRoot) {
    Preconditions.checkNotNull(actionOutputRoot);
    this.actionLogBufferPathGenerator = new ActionLogBufferPathGenerator(actionOutputRoot);
    this.skyframeActionExecutor.setActionLogBufferPathGenerator(actionLogBufferPathGenerator);
  }

  @VisibleForTesting
  void setRemoteOutputsMode(RemoteOutputsMode remoteOutputsMode) {
    PrecomputedValue.REMOTE_OUTPUTS_MODE.set(injectable(), remoteOutputsMode);
  }

  private void setRemoteExecutionEnabled(boolean enabled) {
    PrecomputedValue.REMOTE_EXECUTION_ENABLED.set(injectable(), enabled);
  }

  /** Called each time there is a new top-level host configuration. */
  protected void updateTopLevelHostConfiguration(BuildConfiguration topLevelHostConfiguration) {}

  /**
   * Asks the Skyframe evaluator to build the value for BuildConfigurationCollection and returns the
   * result.
   */
  // TODO(ulfjack): Remove this legacy method after switching to the Skyframe-based implementation.
  public BuildConfigurationCollection createConfigurations(
      ExtendedEventHandler eventHandler,
      BuildOptions buildOptions,
      Set<String> multiCpu,
      boolean keepGoing)
      throws InvalidConfigurationException {

    if (configuredTargetProgress != null) {
      configuredTargetProgress.reset();
    }

    List<BuildConfiguration> topLevelTargetConfigs =
        getConfigurations(
            eventHandler,
            PrepareAnalysisPhaseFunction.getTopLevelBuildOptions(buildOptions, multiCpu),
            buildOptions,
            keepGoing);

    BuildConfiguration firstTargetConfig = topLevelTargetConfigs.get(0);

    BuildOptions targetOptions = firstTargetConfig.getOptions();
    BuildOptions hostOptions =
        targetOptions.get(CoreOptions.class).useDistinctHostConfiguration
            ? HostTransition.INSTANCE.patch(targetOptions)
            : targetOptions;
    BuildConfiguration hostConfig = getConfiguration(eventHandler, hostOptions, keepGoing);

    // TODO(gregce): cache invalid option errors in BuildConfigurationFunction, then use a dedicated
    // accessor (i.e. not the event handler) to trigger the exception below.
    ErrorSensingEventHandler nosyEventHandler = new ErrorSensingEventHandler(eventHandler);
    topLevelTargetConfigs.forEach(config -> config.reportInvalidOptions(nosyEventHandler));
    if (nosyEventHandler.hasErrors()) {
      throw new InvalidConfigurationException("Build options are invalid");
    }
    return new BuildConfigurationCollection(topLevelTargetConfigs, hostConfig);
  }

  /**
   * Asks the Skyframe evaluator to build the given artifacts and targets, and to test the given
   * parallel test targets. Additionally, exclusive tests are built together with all the other
   * tests but they are intentionally *not* run since they must be executed separately one-by-one.
   */
  public EvaluationResult<?> buildArtifacts(
      Reporter reporter,
      ResourceManager resourceManager,
      Executor executor,
      Set<Artifact> artifactsToBuild,
      Collection<ConfiguredTarget> targetsToBuild,
      Collection<AspectValue> aspects,
      Set<ConfiguredTarget> parallelTests,
      Set<ConfiguredTarget> exclusiveTests,
      OptionsProvider options,
      ActionCacheChecker actionCacheChecker,
      TopDownActionCache topDownActionCache,
      @Nullable EvaluationProgressReceiver executionProgressReceiver,
      TopLevelArtifactContext topLevelArtifactContext)
      throws InterruptedException {
    checkActive();
    Preconditions.checkState(actionLogBufferPathGenerator != null);

    try (SilentCloseable c =
        Profiler.instance().profile("skyframeActionExecutor.prepareForExecution")) {
      skyframeActionExecutor.prepareForExecution(
          reporter, executor, options, actionCacheChecker, topDownActionCache, outputService);
    }

    resourceManager.resetResourceUsage();
    try {
      progressReceiver.executionProgressReceiver = executionProgressReceiver;
      Iterable<TargetCompletionValue.TargetCompletionKey> targetKeys =
          TargetCompletionValue.keys(
              targetsToBuild, topLevelArtifactContext, Sets.union(parallelTests, exclusiveTests));
      Iterable<SkyKey> aspectKeys = AspectCompletionValue.keys(aspects, topLevelArtifactContext);
      Iterable<SkyKey> testKeys =
          TestCompletionValue.keys(
              parallelTests, topLevelArtifactContext, /*exclusiveTesting=*/ false);
      EvaluationContext evaluationContext =
          EvaluationContext.newBuilder()
              .setKeepGoing(options.getOptions(KeepGoingOption.class).keepGoing)
              .setNumThreads(options.getOptions(BuildRequestOptions.class).jobs)
              .setUseForkJoinPool(options.getOptions(BuildRequestOptions.class).useForkJoinPool)
              .setEventHander(reporter)
              .setExecutionPhase()
              .build();
      return buildDriver.evaluate(
          Iterables.concat(Artifact.keys(artifactsToBuild), targetKeys, aspectKeys, testKeys),
          evaluationContext);
    } finally {
      progressReceiver.executionProgressReceiver = null;
      // Also releases thread locks.
      resourceManager.resetResourceUsage();
      skyframeActionExecutor.executionOver();
      actionExecutionFunction.complete(reporter);
    }
  }

  /** Asks the Skyframe evaluator to run a single exclusive test. */
  public EvaluationResult<?> runExclusiveTest(
      Reporter reporter,
      ResourceManager resourceManager,
      Executor executor,
      ConfiguredTarget exclusiveTest,
      OptionsProvider options,
      ActionCacheChecker actionCacheChecker,
      TopDownActionCache topDownActionCache,
      @Nullable EvaluationProgressReceiver executionProgressReceiver,
      TopLevelArtifactContext topLevelArtifactContext)
      throws InterruptedException {
    checkActive();
    Preconditions.checkState(actionLogBufferPathGenerator != null);

    try (SilentCloseable c =
        Profiler.instance().profile("skyframeActionExecutor.prepareForExecution")) {
      skyframeActionExecutor.prepareForExecution(
          reporter, executor, options, actionCacheChecker, topDownActionCache, outputService);
    }

    resourceManager.resetResourceUsage();
    try {
      Iterable<SkyKey> testKeys =
          TestCompletionValue.keys(
              ImmutableSet.of(exclusiveTest), topLevelArtifactContext, /*exclusiveTesting=*/ true);
      return evaluate(
          testKeys,
          /*keepGoing=*/ options.getOptions(KeepGoingOption.class).keepGoing,
          /*numThreads=*/ options.getOptions(BuildRequestOptions.class).jobs,
          reporter);
    } finally {
      progressReceiver.executionProgressReceiver = null;
      // Also releases thread locks.
      resourceManager.resetResourceUsage();
      skyframeActionExecutor.executionOver();
      actionExecutionFunction.complete(reporter);
    }
  }

  @VisibleForTesting
  public void prepareBuildingForTestingOnly(
      Reporter reporter,
      Executor executor,
      OptionsProvider options,
      ActionCacheChecker checker,
      TopDownActionCache topDownActionCache) {
    skyframeActionExecutor.prepareForExecution(
        reporter, executor, options, checker, topDownActionCache, outputService);
  }

  EvaluationResult<SkyValue> targetPatterns(
      Iterable<? extends SkyKey> patternSkyKeys,
      int numThreads,
      boolean keepGoing,
      ExtendedEventHandler eventHandler)
      throws InterruptedException {
    checkActive();
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(keepGoing)
            .setNumThreads(numThreads)
            .setEventHander(eventHandler)
            .setUseForkJoinPool(true)
            .build();
    return buildDriver.evaluate(patternSkyKeys, evaluationContext);
  }

  @VisibleForTesting
  public BuildOptions getDefaultBuildOptions() {
    return defaultBuildOptions;
  }

  /**
   * Returns the {@link ConfiguredTargetAndData}s corresponding to the given keys.
   *
   * <p>For use for legacy support and tests calling through {@code BuildView} only.
   *
   * <p>If a requested configured target is in error, the corresponding value is omitted from the
   * returned list.
   */
  @ThreadSafety.ThreadSafe
  public ImmutableList<ConfiguredTargetAndData> getConfiguredTargetsForTesting(
      ExtendedEventHandler eventHandler,
      BuildConfiguration originalConfig,
      Iterable<Dependency> keys)
      throws TransitionException, InvalidConfigurationException {
    return getConfiguredTargetMapForTesting(eventHandler, originalConfig, keys).values().asList();
  }

  /**
   * Returns the {@link ConfiguredTargetAndData}s corresponding to the given keys.
   *
   * <p>For use for legacy support and tests calling through {@code BuildView} only.
   *
   * <p>If a requested configured target is in error, the corresponding value is omitted from the
   * returned list.
   */
  @ThreadSafety.ThreadSafe
  public ImmutableList<ConfiguredTargetAndData> getConfiguredTargetsForTesting(
      ExtendedEventHandler eventHandler,
      BuildConfigurationValue.Key originalConfig,
      Iterable<Dependency> keys)
      throws TransitionException, InvalidConfigurationException {
    return getConfiguredTargetMapForTesting(eventHandler, originalConfig, keys).values().asList();
  }

  /**
   * Returns a map from {@link Dependency} inputs to the {@link ConfiguredTargetAndData}s
   * corresponding to those dependencies.
   *
   * <p>For use for legacy support and tests calling through {@code BuildView} only.
   *
   * <p>If a requested configured target is in error, the corresponding value is omitted from the
   * returned list.
   */
  @ThreadSafety.ThreadSafe
  public ImmutableMultimap<Dependency, ConfiguredTargetAndData> getConfiguredTargetMapForTesting(
      ExtendedEventHandler eventHandler,
      BuildConfigurationValue.Key originalConfig,
      Iterable<Dependency> keys)
      throws TransitionException, InvalidConfigurationException {
    return getConfiguredTargetMapForTesting(
        eventHandler, getConfiguration(eventHandler, originalConfig), keys);
  }

  /**
   * Returns a map from {@link Dependency} inputs to the {@link ConfiguredTargetAndData}s
   * corresponding to those dependencies.
   *
   * <p>For use for legacy support and tests calling through {@code BuildView} only.
   *
   * <p>If a requested configured target is in error, the corresponding value is omitted from the
   * returned list except...
   */
  @ThreadSafety.ThreadSafe
  private ImmutableMultimap<Dependency, ConfiguredTargetAndData> getConfiguredTargetMapForTesting(
      ExtendedEventHandler eventHandler,
      BuildConfiguration originalConfig,
      Iterable<Dependency> keys)
      throws InvalidConfigurationException {
    checkActive();

    Multimap<Dependency, BuildConfiguration> configs;
    if (originalConfig != null) {
      configs =
          getConfigurations(eventHandler, originalConfig.getOptions(), keys).getConfigurationMap();
    } else {
      configs = ArrayListMultimap.<Dependency, BuildConfiguration>create();
      for (Dependency key : keys) {
        configs.put(key, null);
      }
    }

    final List<SkyKey> skyKeys = new ArrayList<>();
    for (Dependency key : keys) {
      if (!configs.containsKey(key)) {
        // If we couldn't compute a configuration for this target, the target was in error (e.g.
        // it couldn't be loaded). Exclude it from the results.
        continue;
      }
      for (BuildConfiguration depConfig : configs.get(key)) {
        skyKeys.add(ConfiguredTargetValue.key(key.getLabel(), depConfig));
        for (AspectDescriptor aspectDescriptor : key.getAspects().getAllAspects()) {
          skyKeys.add(
              AspectValue.createAspectKey(key.getLabel(), depConfig, aspectDescriptor, depConfig));
        }
      }
      skyKeys.add(PackageValue.key(key.getLabel().getPackageIdentifier()));
    }

    EvaluationResult<SkyValue> result = evaluateSkyKeys(eventHandler, skyKeys);

    ImmutableMultimap.Builder<Dependency, ConfiguredTargetAndData> cts =
        ImmutableMultimap.builder();

    // Logic copied from ConfiguredTargetFunction#computeDependencies.
    Set<SkyKey> aliasPackagesToFetch = new HashSet<>();
    List<Dependency> aliasKeysToRedo = new ArrayList<>();
    EvaluationResult<SkyValue> aliasPackageValues = null;
    Iterable<Dependency> keysToProcess = keys;
    for (int i = 0; i < 2; i++) {
      DependentNodeLoop:
      for (Dependency key : keysToProcess) {
        if (!configs.containsKey(key)) {
          // If we couldn't compute a configuration for this target, the target was in error (e.g.
          // it couldn't be loaded). Exclude it from the results.
          continue;
        }
        for (BuildConfiguration depConfig : configs.get(key)) {
          SkyKey configuredTargetKey = ConfiguredTargetValue.key(key.getLabel(), depConfig);
          if (result.get(configuredTargetKey) == null) {
            continue;
          }

          ConfiguredTarget configuredTarget =
              ((ConfiguredTargetValue) result.get(configuredTargetKey)).getConfiguredTarget();
          Label label = configuredTarget.getLabel();
          SkyKey packageKey = PackageValue.key(label.getPackageIdentifier());
          PackageValue packageValue;
          if (i == 0) {
            packageValue = (PackageValue) result.get(packageKey);
            if (packageValue == null) {
              aliasPackagesToFetch.add(packageKey);
              aliasKeysToRedo.add(key);
              continue;
            }
          } else {
            packageValue =
                (PackageValue)
                    Preconditions.checkNotNull(aliasPackageValues.get(packageKey), packageKey);
          }
          List<ConfiguredAspect> configuredAspects = new ArrayList<>();

          for (AspectDescriptor aspectDescriptor : key.getAspects().getAllAspects()) {
            SkyKey aspectKey =
                AspectValue.createAspectKey(key.getLabel(), depConfig, aspectDescriptor, depConfig);
            if (result.get(aspectKey) == null) {
              continue DependentNodeLoop;
            }

            configuredAspects.add(((AspectValue) result.get(aspectKey)).getConfiguredAspect());
          }

          try {
            ConfiguredTarget mergedTarget =
                MergedConfiguredTarget.of(configuredTarget, configuredAspects);
            BuildConfigurationValue.Key configKey = mergedTarget.getConfigurationKey();
            BuildConfiguration resolvedConfig = depConfig;
            if (configKey == null) {
              // Unfortunately, it's possible to get a configured target with a null configuration
              // when depConfig is non-null, so we need to explicitly override it in that case.
              resolvedConfig = null;
            } else if (!configKey.equals(BuildConfigurationValue.key(depConfig))) {
              // Retroactive trimming may change the configuration associated with the dependency.
              // If it does, we need to get that instance.
              // TODO(b/140632978): doing these individually instead of doing them all at once may
              // end up being wasteful use of Skyframe. Although these configurations are guaranteed
              // to be in the Skyframe cache (because the dependency would have had to retrieve
              // them to be created in the first place), looking them up repeatedly may be slower
              // than just keeping a local cache and assigning the same configuration to all the
              // CTs which need it. Profile this and see if there's a better way.
              if (!depConfig.trimConfigurationsRetroactively()) {
                throw new AssertionError(
                    "Loading configurations mid-dependency resolution should ONLY happen when "
                        + "retroactive trimming is enabled.");
              }
              resolvedConfig = getConfiguration(eventHandler, mergedTarget.getConfigurationKey());
            }
            cts.put(
                key,
                new ConfiguredTargetAndData(
                    mergedTarget,
                    packageValue.getPackage().getTarget(configuredTarget.getLabel().getName()),
                    resolvedConfig));

          } catch (DuplicateException | NoSuchTargetException e) {
            throw new IllegalStateException(
                String.format("Error creating %s", configuredTarget.getLabel()), e);
          }
        }
      }
      if (aliasKeysToRedo.isEmpty()) {
        break;
      }
      aliasPackageValues = evaluateSkyKeys(eventHandler, aliasPackagesToFetch);
      keysToProcess = aliasKeysToRedo;
    }
    Supplier<Map<BuildConfigurationValue.Key, BuildConfiguration>> configurationLookupSupplier =
        () ->
            configs.values().stream()
                .collect(
                    Collectors.toMap(
                        BuildConfigurationValue::key, java.util.function.Function.identity()));
    // We ignore the return value here because tests effectively run with --keep_going, and the
    // loading-phase-error bit is only needed if we're constructing a SkyframeAnalysisResult.
    SkyframeBuildView.processErrors(
        result,
        configurationLookupSupplier,
        this,
        eventHandler,
        /*keepGoing=*/ true,
        /*eventBus=*/ null);
    return cts.build();
  }

  /**
   * Returns the configuration corresponding to the given set of build options. Should not be used
   * in a world with trimmed configurations.
   *
   * @throws InvalidConfigurationException if the build options produces an invalid configuration
   */
  @Deprecated
  public BuildConfiguration getConfiguration(
      ExtendedEventHandler eventHandler, BuildOptions options, boolean keepGoing)
      throws InvalidConfigurationException {
    return Iterables.getOnlyElement(
        getConfigurations(eventHandler, ImmutableList.of(options), options, keepGoing));
  }

  @VisibleForTesting
  public BuildConfiguration getConfiguration(
      ExtendedEventHandler eventHandler, BuildConfigurationValue.Key configurationKey) {
    if (configurationKey == null) {
      return null;
    }
    return ((BuildConfigurationValue)
            evaluateSkyKeys(eventHandler, ImmutableList.of(configurationKey)).get(configurationKey))
        .getConfiguration();
  }

  public Map<BuildConfigurationValue.Key, BuildConfiguration> getConfigurations(
      ExtendedEventHandler eventHandler, Collection<BuildConfigurationValue.Key> keys) {
    EvaluationResult<SkyValue> evaluationResult = evaluateSkyKeys(eventHandler, keys);
    return keys.stream()
        .collect(
            Collectors.toMap(
                java.util.function.Function.identity(),
                (key) -> ((BuildConfigurationValue) evaluationResult.get(key)).getConfiguration()));
  }
  /**
   * Returns the configurations corresponding to the given sets of build options. Output order is
   * the same as input order.
   *
   * @throws InvalidConfigurationException if any build options produces an invalid configuration
   */
  // TODO(ulfjack): Remove this legacy method after switching to the Skyframe-based implementation.
  private List<BuildConfiguration> getConfigurations(
      ExtendedEventHandler eventHandler,
      List<BuildOptions> optionsList,
      BuildOptions referenceBuildOptions,
      boolean keepGoing)
      throws InvalidConfigurationException {
    Preconditions.checkArgument(!Iterables.isEmpty(optionsList));

    // Prepare the Skyframe inputs.
    // TODO(gregce): support trimmed configs.
    ImmutableSortedSet<Class<? extends BuildConfiguration.Fragment>> allFragments =
        ruleClassProvider.getConfigurationFragments().stream()
            .map(factory -> factory.creates())
            .collect(
                ImmutableSortedSet.toImmutableSortedSet(BuildConfiguration.lexicalFragmentSorter));

    PlatformMappingValue platformMappingValue =
        getPlatformMappingValue(eventHandler, referenceBuildOptions);

    ImmutableList.Builder<SkyKey> configSkyKeysBuilder = ImmutableList.builder();
    for (BuildOptions options : optionsList) {
      configSkyKeysBuilder.add(toConfigurationKey(platformMappingValue, allFragments, options));
    }

    ImmutableList<SkyKey> configSkyKeys = configSkyKeysBuilder.build();

    // Skyframe-evaluate the configurations and throw errors if any.
    EvaluationResult<SkyValue> evalResult = evaluateSkyKeys(eventHandler, configSkyKeys, keepGoing);
    if (evalResult.hasError()) {
      Map.Entry<SkyKey, ErrorInfo> firstError = Iterables.get(evalResult.errorMap().entrySet(), 0);
      ErrorInfo error = firstError.getValue();
      Throwable e = error.getException();
      // Wrap loading failed exceptions
      if (e instanceof NoSuchThingException) {
        e = new InvalidConfigurationException(e);
      } else if (e == null && !Iterables.isEmpty(error.getCycleInfo())) {
        getCyclesReporter().reportCycles(error.getCycleInfo(), firstError.getKey(), eventHandler);
        e =
            new InvalidConfigurationException(
                "cannot load build configuration because of this cycle");
      }
      if (e != null) {
        Throwables.throwIfInstanceOf(e, InvalidConfigurationException.class);
      }
      throw new IllegalStateException("Unknown error during configuration creation evaluation", e);
    }

    // Prepare and return the results.
    return configSkyKeys.stream()
        .map(key -> ((BuildConfigurationValue) evalResult.get(key)).getConfiguration())
        .collect(ImmutableList.toImmutableList());
  }

  /**
   * Retrieves the configurations needed for the given deps. If {@link
   * CoreOptions#trimConfigurations()} is true, trims their fragments to only those needed by their
   * transitive closures. Else unconditionally includes all fragments.
   *
   * <p>Skips targets with loading phase errors.
   */
  // Keep this in sync with {@link PrepareAnalysisPhaseFunction#getConfigurations}.
  // TODO(ulfjack): Remove this legacy method after switching to the Skyframe-based implementation.
  public ConfigurationsResult getConfigurations(
      ExtendedEventHandler eventHandler, BuildOptions fromOptions, Iterable<Dependency> keys)
      throws InvalidConfigurationException {
    ConfigurationsResult.Builder builder = ConfigurationsResult.newBuilder();
    Set<Dependency> depsToEvaluate = new HashSet<>();

    ImmutableSortedSet<Class<? extends BuildConfiguration.Fragment>> allFragments = null;
    if (useUntrimmedConfigs(fromOptions)) {
      allFragments = ruleClassProvider.getAllFragments();
    }

    // Get the fragments needed for dynamic configuration nodes.
    final List<SkyKey> transitiveFragmentSkyKeys = new ArrayList<>();
    Map<Label, ImmutableSortedSet<Class<? extends BuildConfiguration.Fragment>>> fragmentsMap =
        new HashMap<>();
    Set<Label> labelsWithErrors = new HashSet<>();
    for (Dependency key : keys) {
      if (key.hasExplicitConfiguration()) {
        builder.put(key, key.getConfiguration());
      } else if (useUntrimmedConfigs(fromOptions)) {
        fragmentsMap.put(key.getLabel(), allFragments);
      } else {
        depsToEvaluate.add(key);
        transitiveFragmentSkyKeys.add(TransitiveTargetKey.of(key.getLabel()));
      }
    }
    EvaluationResult<SkyValue> fragmentsResult =
        evaluateSkyKeys(eventHandler, transitiveFragmentSkyKeys, /*keepGoing=*/ true);
    for (Map.Entry<SkyKey, ErrorInfo> entry : fragmentsResult.errorMap().entrySet()) {
      reportCycles(eventHandler, entry.getValue().getCycleInfo(), entry.getKey());
    }
    for (Dependency key : keys) {
      if (!depsToEvaluate.contains(key)) {
        // No fragments to compute here.
      } else if (fragmentsResult.getError(TransitiveTargetKey.of(key.getLabel())) != null) {
        labelsWithErrors.add(key.getLabel());
        builder.setHasError();
      } else {
        TransitiveTargetValue ttv =
            (TransitiveTargetValue) fragmentsResult.get(TransitiveTargetKey.of(key.getLabel()));
        fragmentsMap.put(
            key.getLabel(),
            ImmutableSortedSet.copyOf(
                BuildConfiguration.lexicalFragmentSorter,
                ttv.getTransitiveConfigFragments().toSet()));
      }
    }

    PlatformMappingValue platformMappingValue = getPlatformMappingValue(eventHandler, fromOptions);

    // Now get the configurations.
    final List<SkyKey> configSkyKeys = new ArrayList<>();
    for (Dependency key : keys) {
      if (labelsWithErrors.contains(key.getLabel()) || key.hasExplicitConfiguration()) {
        continue;
      }
      ImmutableSortedSet<Class<? extends BuildConfiguration.Fragment>> depFragments =
          fragmentsMap.get(key.getLabel());
      if (depFragments != null) {
        ConfigurationTransition transition = key.getTransition();
        if (transition == NullTransition.INSTANCE) {
          continue;
        }
        List<BuildOptions> toOptions = Collections.singletonList(fromOptions);
        try {
          Map<PackageValue.Key, PackageValue> buildSettingPackages =
              getBuildSettingPackages(transition, eventHandler);
          toOptions =
              ConfigurationResolver.applyTransition(
                  fromOptions, transition, buildSettingPackages, eventHandler);
          StarlarkTransition.replayEvents(eventHandler, transition);
        } catch (TransitionException e) {
          eventHandler.handle(Event.error(e.getMessage()));
          builder.setHasError();
          continue;
        }
        for (BuildOptions toOption : toOptions) {
          configSkyKeys.add(toConfigurationKey(platformMappingValue, depFragments, toOption));
        }
      }
    }
    EvaluationResult<SkyValue> configsResult =
        evaluateSkyKeys(eventHandler, configSkyKeys, /*keepGoing=*/ true);
    for (Dependency key : keys) {
      if (labelsWithErrors.contains(key.getLabel()) || key.hasExplicitConfiguration()) {
        continue;
      }
      ImmutableSortedSet<Class<? extends BuildConfiguration.Fragment>> depFragments =
          fragmentsMap.get(key.getLabel());
      if (depFragments != null) {
        if (key.getTransition() == NullTransition.INSTANCE) {
          builder.put(key, null);
          continue;
        }
        List<BuildOptions> toOptions = Collections.singletonList(fromOptions);
        try {
          Map<PackageValue.Key, PackageValue> buildSettingPackages =
              getBuildSettingPackages(key.getTransition(), eventHandler);
          toOptions =
              ConfigurationResolver.applyTransition(
                  fromOptions, key.getTransition(), buildSettingPackages, eventHandler);
        } catch (TransitionException e) {
          eventHandler.handle(Event.error(e.getMessage()));
          builder.setHasError();
          continue;
        }
        for (BuildOptions toOption : toOptions) {
          BuildConfigurationValue.Key configKey =
              toConfigurationKey(platformMappingValue, depFragments, toOption);
          BuildConfigurationValue configValue =
              ((BuildConfigurationValue) configsResult.get(configKey));
          // configValue will be null here if there was an exception thrown during configuration
          // creation. This will be reported elsewhere.
          if (configValue != null) {
            builder.put(key, configValue.getConfiguration());
          }
        }
      }
    }
    return builder.build();
  }

  /**
   * The result of {@link #getConfigurations(ExtendedEventHandler, BuildOptions, Iterable)} which
   * also registers if an error was recorded.
   */
  public static class ConfigurationsResult {
    private final Multimap<Dependency, BuildConfiguration> configurations;
    private final boolean hasError;

    private ConfigurationsResult(
        Multimap<Dependency, BuildConfiguration> configurations, boolean hasError) {
      this.configurations = configurations;
      this.hasError = hasError;
    }

    public boolean hasError() {
      return hasError;
    }

    public Multimap<Dependency, BuildConfiguration> getConfigurationMap() {
      return configurations;
    }

    public static Builder newBuilder() {
      return new Builder();
    }

    /** Builder for {@link ConfigurationsResult} */
    public static class Builder {
      private final Multimap<Dependency, BuildConfiguration> configurations =
          ArrayListMultimap.<Dependency, BuildConfiguration>create();
      private boolean hasError = false;

      void put(Dependency key, BuildConfiguration value) {
        configurations.put(key, value);
      }

      void setHasError() {
        this.hasError = true;
      }

      ConfigurationsResult build() {
        return new ConfigurationsResult(configurations, hasError);
      }
    }
  }

  private PlatformMappingValue getPlatformMappingValue(
      ExtendedEventHandler eventHandler, BuildOptions referenceBuildOptions)
      throws InvalidConfigurationException {
    PathFragment platformMappingPath =
        referenceBuildOptions.get(PlatformOptions.class).platformMappings;

    PlatformMappingValue.Key platformMappingKey =
        PlatformMappingValue.Key.create(platformMappingPath);
    EvaluationResult<SkyValue> evaluationResult =
        evaluateSkyKeys(eventHandler, ImmutableSet.of(platformMappingKey));
    if (evaluationResult.hasError()) {
      throw new InvalidConfigurationException(evaluationResult.getError().getException());
    }
    return (PlatformMappingValue) evaluationResult.get(platformMappingKey);
  }

  private BuildConfigurationValue.Key toConfigurationKey(
      PlatformMappingValue platformMappingValue,
      ImmutableSortedSet<Class<? extends BuildConfiguration.Fragment>> depFragments,
      BuildOptions toOption)
      throws InvalidConfigurationException {
    try {
      return BuildConfigurationValue.keyWithPlatformMapping(
          platformMappingValue,
          defaultBuildOptions,
          depFragments,
          BuildOptions.diffForReconstruction(defaultBuildOptions, toOption));
    } catch (OptionsParsingException e) {
      throw new InvalidConfigurationException(e);
    }
  }

  /** Keep in sync with {@link StarlarkTransition#getBuildSettingPackages} */
  private Map<PackageValue.Key, PackageValue> getBuildSettingPackages(
      ConfigurationTransition transition, ExtendedEventHandler eventHandler)
      throws TransitionException {
    HashMap<PackageValue.Key, PackageValue> buildSettingPackages = new HashMap<>();
    // This happens before cycle detection so keep track of all seen build settings to ensure
    // we don't get stuck in endless loops (e.g. //alias1->//alias2 && //alias2->alias1)
    Set<Label> allSeenBuildSettings = new HashSet<>();
    ImmutableSet<Label> unverifiedBuildSettings =
        StarlarkTransition.getAllBuildSettings(transition);
    while (!unverifiedBuildSettings.isEmpty()) {
      for (Label buildSetting : unverifiedBuildSettings) {
        if (!allSeenBuildSettings.add(buildSetting)) {
          throw new TransitionException(
              String.format(
                  "Error with aliased build settings related to '%s'. Either your aliases form a"
                      + " dependency cycle or you're attempting to set both an alias its actual"
                      + " target in the same transition.",
                  buildSetting));
        }
      }
      ImmutableSet<PackageValue.Key> buildSettingPackageKeys =
          StarlarkTransition.getPackageKeysFromLabels(unverifiedBuildSettings);
      EvaluationResult<SkyValue> newlyLoaded =
          evaluateSkyKeys(eventHandler, buildSettingPackageKeys, true);
      if (newlyLoaded.hasError()) {
        throw new TransitionException(
            new NoSuchPackageException(
                ((PackageValue.Key) newlyLoaded.getError().getRootCauseOfException()).argument(),
                "Unable to find build setting package",
                newlyLoaded.getError().getException()));
      }
      buildSettingPackageKeys.forEach(
          k -> buildSettingPackages.put(k, (PackageValue) newlyLoaded.get(k)));
      unverifiedBuildSettings =
          StarlarkTransition.verifyBuildSettingsAndGetAliases(
              buildSettingPackages, unverifiedBuildSettings);
    }
    return buildSettingPackages;
  }

  /**
   * Returns whether configurations should trim their fragments to only those needed by targets and
   * their transitive dependencies.
   */
  private static boolean useUntrimmedConfigs(BuildOptions options) {
    return options.get(CoreOptions.class).configsMode == CoreOptions.ConfigsMode.NOTRIM;
  }

  /**
   * Evaluates the given sky keys, blocks, and returns their evaluation results. Fails fast on the
   * first evaluation error.
   */
  private EvaluationResult<SkyValue> evaluateSkyKeys(
      final ExtendedEventHandler eventHandler, final Iterable<? extends SkyKey> skyKeys) {
    return evaluateSkyKeys(eventHandler, skyKeys, false);
  }

  /**
   * Evaluates the given sky keys, blocks, and returns their evaluation results. Enables/disables
   * "keep going" on evaluation errors as specified.
   */
  EvaluationResult<SkyValue> evaluateSkyKeys(
      final ExtendedEventHandler eventHandler,
      final Iterable<? extends SkyKey> skyKeys,
      final boolean keepGoing) {
    EvaluationResult<SkyValue> result;
    try {
      result =
          callUninterruptibly(
              new Callable<EvaluationResult<SkyValue>>() {
                @Override
                public EvaluationResult<SkyValue> call() throws Exception {
                  synchronized (valueLookupLock) {
                    try {
                      skyframeBuildView.enableAnalysis(true);
                      return evaluate(
                          skyKeys, keepGoing, /*numThreads=*/ DEFAULT_THREAD_COUNT, eventHandler);
                    } finally {
                      skyframeBuildView.enableAnalysis(false);
                    }
                  }
                }
              });
    } catch (Exception e) {
      throw new IllegalStateException(e); // Should never happen.
    }
    return result;
  }

  /**
   * Returns a dynamic configuration constructed from the given configuration fragments and build
   * options.
   */
  @VisibleForTesting
  public BuildConfiguration getConfigurationForTesting(
      ExtendedEventHandler eventHandler, FragmentClassSet fragments, BuildOptions options)
      throws InterruptedException, OptionsParsingException, InvalidConfigurationException {
    SkyKey key =
        BuildConfigurationValue.keyWithPlatformMapping(
            getPlatformMappingValue(eventHandler, options),
            defaultBuildOptions,
            fragments,
            BuildOptions.diffForReconstruction(defaultBuildOptions, options));
    BuildConfigurationValue result =
        (BuildConfigurationValue)
            evaluate(
                    ImmutableList.of(key),
                    /*keepGoing=*/ false,
                    /*numThreads=*/ DEFAULT_THREAD_COUNT,
                    eventHandler)
                .get(key);
    return result.getConfiguration();
  }

  /** Returns a particular configured target. */
  @VisibleForTesting
  @Nullable
  public ConfiguredTarget getConfiguredTargetForTesting(
      ExtendedEventHandler eventHandler, Label label, BuildConfiguration configuration)
      throws TransitionException, InvalidConfigurationException {
    return getConfiguredTargetForTesting(eventHandler, label, configuration, NoTransition.INSTANCE);
  }

  /** Returns a particular configured target after applying the given transition. */
  @VisibleForTesting
  @Nullable
  public ConfiguredTarget getConfiguredTargetForTesting(
      ExtendedEventHandler eventHandler,
      Label label,
      BuildConfiguration configuration,
      ConfigurationTransition transition)
      throws TransitionException, InvalidConfigurationException {
    ConfiguredTargetAndData configuredTargetAndData =
        getConfiguredTargetAndDataForTesting(eventHandler, label, configuration, transition);
    return configuredTargetAndData == null ? null : configuredTargetAndData.getConfiguredTarget();
  }

  @VisibleForTesting
  @Nullable
  public ConfiguredTargetAndData getConfiguredTargetAndDataForTesting(
      ExtendedEventHandler eventHandler,
      Label label,
      BuildConfiguration configuration,
      ConfigurationTransition transition)
      throws TransitionException, InvalidConfigurationException {
    return Iterables.getFirst(
        getConfiguredTargetsForTesting(
            eventHandler,
            configuration,
            ImmutableList.of(
                configuration == null
                    ? Dependency.withNullConfiguration(label)
                    : Dependency.withTransitionAndAspects(
                        label, transition, AspectCollection.EMPTY))),
        null);
  }

  @VisibleForTesting
  @Nullable
  public ConfiguredTargetAndData getConfiguredTargetAndDataForTesting(
      ExtendedEventHandler eventHandler, Label label, BuildConfiguration configuration)
      throws TransitionException, InvalidConfigurationException {
    return getConfiguredTargetAndDataForTesting(
        eventHandler, label, configuration, NoTransition.INSTANCE);
  }

  /**
   * Invalidates Skyframe values corresponding to the given set of modified files under the given
   * path entry.
   *
   * <p>May throw an {@link InterruptedException}, which means that no values have been invalidated.
   */
  @VisibleForTesting
  public final void invalidateFilesUnderPathForTesting(
      ExtendedEventHandler eventHandler, ModifiedFileSet modifiedFileSet, Root pathEntry)
      throws InterruptedException {
    if (lastAnalysisDiscarded) {
      // Values were cleared last build, but they couldn't be deleted because they were needed for
      // the execution phase. We can delete them now.
      dropConfiguredTargetsNow(eventHandler);
      lastAnalysisDiscarded = false;
    }
    invalidateFilesUnderPathForTestingImpl(eventHandler, modifiedFileSet, pathEntry);
  }

  @VisibleForTesting
  public final void turnOffSyscallCacheForTesting() {
    syscalls.set(UnixGlob.DEFAULT_SYSCALLS);
  }

  protected abstract void invalidateFilesUnderPathForTestingImpl(
      ExtendedEventHandler eventHandler, ModifiedFileSet modifiedFileSet, Root pathEntry)
      throws InterruptedException;

  /** Invalidates SkyFrame values that may have failed for transient reasons. */
  public abstract void invalidateTransientErrors();

  /** Configures a given set of configured targets. */
  EvaluationResult<ActionLookupValue> configureTargets(
      ExtendedEventHandler eventHandler,
      List<ConfiguredTargetKey> values,
      List<AspectValueKey> aspectKeys,
      boolean keepGoing,
      int numThreads)
      throws InterruptedException {
    checkActive();

    List<SkyKey> keys = new ArrayList<>(ConfiguredTargetValue.keys(values));
    for (AspectValueKey aspectKey : aspectKeys) {
      keys.add(aspectKey);
    }
    eventHandler.post(new ConfigurationPhaseStartedEvent(configuredTargetProgress));
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(keepGoing)
            .setNumThreads(numThreads)
            .setExecutorServiceSupplier(
                () -> NamedForkJoinPool.newNamedPool("skyframe-evaluator", numThreads))
            .setEventHander(eventHandler)
            .build();
    EvaluationResult<ActionLookupValue> result = buildDriver.evaluate(keys, evaluationContext);
    // Get rid of any memory retained by the cache -- all loading is done.
    perBuildSyscallCache.clear();
    return result;
  }

  /**
   * Post-process the targets. Values in the EvaluationResult are known to be transitively
   * error-free from action conflicts.
   */
  public EvaluationResult<PostConfiguredTargetValue> postConfigureTargets(
      ExtendedEventHandler eventHandler,
      List<ConfiguredTargetKey> values,
      boolean keepGoing,
      ImmutableMap<ActionAnalysisMetadata, SkyframeActionExecutor.ConflictException> badActions)
      throws InterruptedException {
    checkActive();
    PrecomputedValue.BAD_ACTIONS.set(injectable(), badActions);
    // Make sure to not run too many analysis threads. This can cause memory thrashing.
    EvaluationResult<PostConfiguredTargetValue> result =
        evaluate(
            PostConfiguredTargetValue.keys(values),
            keepGoing,
            /*numThreads=*/ ResourceUsage.getAvailableProcessors(),
            eventHandler);

    // Remove all post-configured target values immediately for memory efficiency. We are OK with
    // this mini-phase being non-incremental as the failure mode of action conflict is rare.
    memoizingEvaluator.delete(SkyFunctionName.functionIs(SkyFunctions.POST_CONFIGURED_TARGET));

    return result;
  }

  /** Returns a Skyframe-based {@link SkyframeTransitivePackageLoader} implementation. */
  @VisibleForTesting
  public TransitivePackageLoader pkgLoader() {
    checkActive();
    return new SkyframeLabelVisitor(new SkyframeTransitivePackageLoader(), cyclesReporter);
  }

  class SkyframeTransitivePackageLoader {
    /** Loads the specified {@link TransitiveTargetValue}s. */
    EvaluationResult<TransitiveTargetValue> loadTransitiveTargets(
        ExtendedEventHandler eventHandler,
        Iterable<Label> labelsToVisit,
        boolean keepGoing,
        int parallelThreads)
        throws InterruptedException {
      List<SkyKey> valueNames = new ArrayList<>();
      for (Label label : labelsToVisit) {
        valueNames.add(TransitiveTargetKey.of(label));
      }
      EvaluationContext evaluationContext =
          EvaluationContext.newBuilder()
              .setKeepGoing(keepGoing)
              .setNumThreads(parallelThreads)
              .setEventHander(eventHandler)
              .setUseForkJoinPool(true)
              .build();
      return buildDriver.evaluate(valueNames, evaluationContext);
    }
  }

  /**
   * For internal use in queries: performs a graph update to make sure the transitive closure of the
   * specified {@code universeKey} is present in the graph, and returns the {@link
   * EvaluationResult}.
   *
   * <p>The graph update is unconditionally done in keep-going mode, so that the query is guaranteed
   * a complete graph to work on.
   */
  @Override
  public EvaluationResult<SkyValue> prepareAndGet(
      Set<SkyKey> roots, EvaluationContext evaluationContext) throws InterruptedException {
    EvaluationResult<SkyValue> evaluationResult =
        buildDriver.evaluate(roots, evaluationContext.getCopyWithKeepGoing(/*keepGoing=*/ true));
    return evaluationResult;
  }

  /**
   * Get metadata related to the prepareAndGet() lookup. Resulting data is specific to the
   * underlying evaluation implementation.
   */
  public String prepareAndGetMetadata(
      Collection<String> patterns, String offset, OptionsProvider options)
      throws AbruptExitException, InterruptedException {
    return buildDriver.meta(ImmutableList.of(getUniverseKey(patterns, offset)), options);
  }

  @Override
  public SkyKey getUniverseKey(Collection<String> patterns, String offset) {
    return computeUniverseKey(ImmutableList.copyOf(patterns), offset);
  }

  /** Computes the {@link SkyKey} that defines this universe. */
  public static SkyKey computeUniverseKey(Collection<String> patterns, String offset) {
    return PrepareDepsOfPatternsValue.key(ImmutableList.copyOf(patterns), offset);
  }

  /** Returns the generating action of a given artifact ({@code null} if it's a source artifact). */
  private ActionAnalysisMetadata getGeneratingAction(
      ExtendedEventHandler eventHandler, Artifact artifact) throws InterruptedException {
    if (artifact.isSourceArtifact()) {
      return null;
    }

    ActionLookupData generatingActionKey =
        ((Artifact.DerivedArtifact) artifact).getGeneratingActionKey();

    synchronized (valueLookupLock) {
      // Note that this will crash (attempting to run a configured target value builder after
      // analysis) after a failed --nokeep_going analysis in which the configured target that
      // failed was a (transitive) dependency of the configured target that should generate
      // this action. We don't expect callers to query generating actions in such cases.
      EvaluationResult<ActionLookupValue> result =
          evaluate(
              ImmutableList.of(generatingActionKey.getActionLookupKey()),
              /*keepGoing=*/ false,
              /*numThreads=*/ ResourceUsage.getAvailableProcessors(),
              eventHandler);
      if (result.hasError()) {
        return null;
      }
      ActionLookupValue actionLookupValue = result.get(generatingActionKey.getActionLookupKey());
      return actionLookupValue.getActions().get(generatingActionKey.getActionIndex());
    }
  }

  /**
   * Returns an action graph.
   *
   * <p>For legacy compatibility only.
   */
  public ActionGraph getActionGraph(final ExtendedEventHandler eventHandler) {
    return new ActionGraph() {
      @Override
      public ActionAnalysisMetadata getGeneratingAction(final Artifact artifact) {
        try {
          return callUninterruptibly(
              new Callable<ActionAnalysisMetadata>() {
                @Override
                public ActionAnalysisMetadata call() throws InterruptedException {
                  return SkyframeExecutor.this.getGeneratingAction(eventHandler, artifact);
                }
              });
        } catch (Exception e) {
          throw new IllegalStateException(
              "Error getting generating action: " + artifact.prettyPrint(), e);
        }
      }
    };
  }

  public PackageManager getPackageManager() {
    return packageManager;
  }

  @VisibleForTesting
  public TargetPatternPreloader newTargetPatternPreloader() {
    return new SkyframeTargetPatternEvaluator(this);
  }

  public ActionKeyContext getActionKeyContext() {
    return actionKeyContext;
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
    Package getPackage(ExtendedEventHandler eventHandler, PackageIdentifier pkgName)
        throws InterruptedException, NoSuchPackageException {
      synchronized (valueLookupLock) {
        SkyKey key = PackageValue.key(pkgName);
        // Any call to this method post-loading phase should either be error-free or be in a
        // keep_going build, since otherwise the build would have failed during loading. Thus
        // we set keepGoing=true unconditionally.
        EvaluationResult<PackageValue> result =
            evaluate(
                ImmutableList.of(key),
                /*keepGoing=*/ true,
                /*numThreads=*/ DEFAULT_THREAD_COUNT,
                eventHandler);
        ErrorInfo error = result.getError(key);
        if (error != null) {
          if (!Iterables.isEmpty(error.getCycleInfo())) {
            reportCycles(eventHandler, result.getError().getCycleInfo(), key);
            // This can only happen if a package is freshly loaded outside of the target parsing
            // or loading phase
            throw new BuildFileContainsErrorsException(
                pkgName, "Cycle encountered while loading package " + pkgName);
          }
          Throwable e = error.getException();
          // PackageFunction should be catching, swallowing, and rethrowing all transitive
          // errors as NoSuchPackageExceptions or constructing packages with errors, since we're in
          // keep_going mode.
          Throwables.propagateIfInstanceOf(e, NoSuchPackageException.class);
          throw new IllegalStateException(
              "Unexpected Exception type from PackageValue for '"
                  + pkgName
                  + "'' with root causes: "
                  + Iterables.toString(error.getRootCauses()),
              e);
        }
        return result.get(key).getPackage();
      }
    }

    /** Returns whether the given package should be consider deleted and thus should be ignored. */
    public boolean isPackageDeleted(PackageIdentifier packageName) {
      Preconditions.checkState(
          !packageName.getRepository().isDefault(), "package must be absolute: %s", packageName);
      return deletedPackages.get().contains(packageName);
    }
  }

  @VisibleForTesting
  public MemoizingEvaluator getEvaluatorForTesting() {
    return memoizingEvaluator;
  }

  @VisibleForTesting
  public FileSystem getFileSystemForTesting() {
    return fileSystem;
  }

  @VisibleForTesting
  public RuleClassProvider getRuleClassProviderForTesting() {
    return ruleClassProvider;
  }

  @VisibleForTesting
  public PackageFactory getPackageFactoryForTesting() {
    return pkgFactory;
  }

  @VisibleForTesting
  public Package.Builder.Helper getPackageBuilderHelperForTesting() {
    return pkgFactory.getPackageBuilderHelperForTesting();
  }

  @VisibleForTesting
  public BlazeDirectories getBlazeDirectoriesForTesting() {
    return directories;
  }

  @VisibleForTesting
  ActionExecutionStatusReporter getActionExecutionStatusReporterForTesting() {
    return statusReporterRef.get();
  }

  /**
   * Initializes and syncs the graph with the given options, readying it for the next evaluation.
   */
  public void sync(
      ExtendedEventHandler eventHandler,
      PackageCacheOptions packageCacheOptions,
      PathPackageLocator pathPackageLocator,
      StarlarkSemanticsOptions starlarkSemanticsOptions,
      UUID commandId,
      Map<String, String> clientEnv,
      TimestampGranularityMonitor tsgm,
      OptionsProvider options)
      throws InterruptedException, AbruptExitException {
    getActionEnvFromOptions(options.getOptions(CoreOptions.class));
    setRepoEnv(options.getOptions(CoreOptions.class));
    RemoteOptions remoteOptions = options.getOptions(RemoteOptions.class);
    setRemoteOutputsMode(
        remoteOptions != null
            ? remoteOptions.remoteOutputsMode
            // If no value is specified then set it to some value so that it's not null.
            : RemoteOutputsMode.ALL);
    try {
      setRemoteDefaultPlatformProperties(
          remoteOptions != null
              ? remoteOptions.getRemoteDefaultExecProperties()
              : ImmutableMap.of());
    } catch (UserExecException e) {
      throw new AbruptExitException(e.getMessage(), ExitCode.COMMAND_LINE_ERROR, e);
    }
    setRemoteExecutionEnabled(remoteOptions != null && remoteOptions.isRemoteExecutionEnabled());
    syncPackageLoading(
        packageCacheOptions,
        pathPackageLocator,
        starlarkSemanticsOptions,
        commandId,
        clientEnv,
        tsgm,
        options);
    if (lastAnalysisDiscarded) {
      dropConfiguredTargetsNow(eventHandler);
      lastAnalysisDiscarded = false;
    }
  }

  /**
   * Updates the nestedset size threshold if the flag value changed.
   *
   * @return whether an update was made.
   */
  protected static boolean nestedSetAsSkyKeyThresholdUpdatedAndReset(OptionsProvider options) {
    BuildRequestOptions buildRequestOptions = options.getOptions(BuildRequestOptions.class);
    if (buildRequestOptions == null) {
      return false;
    }
    return ArtifactNestedSetFunction.sizeThresholdUpdatedTo(
        buildRequestOptions.nestedSetAsSkyKeyThreshold);
  }

  protected void syncPackageLoading(
      PackageCacheOptions packageCacheOptions,
      PathPackageLocator pathPackageLocator,
      StarlarkSemanticsOptions starlarkSemanticsOptions,
      UUID commandId,
      Map<String, String> clientEnv,
      TimestampGranularityMonitor tsgm,
      OptionsProvider options)
      throws AbruptExitException {
    try (SilentCloseable c = Profiler.instance().profile("preparePackageLoading")) {
      preparePackageLoading(
          pathPackageLocator,
          packageCacheOptions,
          starlarkSemanticsOptions,
          commandId,
          clientEnv,
          tsgm);
    }
    try (SilentCloseable c = Profiler.instance().profile("setDeletedPackages")) {
      setDeletedPackages(packageCacheOptions.getDeletedPackages());
    }

    incrementalBuildMonitor = new SkyframeIncrementalBuildMonitor();
    invalidateTransientErrors();
  }

  private void setRemoteDefaultPlatformProperties(
      Map<String, String> remoteDefaultPlatformProperties) {
    PrecomputedValue.REMOTE_DEFAULT_PLATFORM_PROPERTIES.set(
        injectable(), remoteDefaultPlatformProperties);
  }

  private void getActionEnvFromOptions(CoreOptions opt) {
    // ImmutableMap does not support null values, so use a LinkedHashMap instead.
    LinkedHashMap<String, String> actionEnvironment = new LinkedHashMap<>();
    if (opt != null) {
      for (Map.Entry<String, String> v : opt.actionEnvironment) {
        actionEnvironment.put(v.getKey(), v.getValue());
      }
    }
    setActionEnv(actionEnvironment);
  }

  @VisibleForTesting
  public void setActionEnv(Map<String, String> actionEnv) {
    PrecomputedValue.ACTION_ENV.set(injectable(), actionEnv);
  }

  private void setRepoEnv(CoreOptions opt) {
    LinkedHashMap<String, String> repoEnv = new LinkedHashMap<>();
    if (opt != null) {
      for (Map.Entry<String, String> v : opt.repositoryEnvironment) {
        repoEnv.put(v.getKey(), v.getValue());
      }
    }
    PrecomputedValue.REPO_ENV.set(injectable(), repoEnv);
  }

  public PathPackageLocator createPackageLocator(
      ExtendedEventHandler eventHandler, List<String> packagePaths, Path workingDirectory) {
    return PathPackageLocator.create(
        directories.getOutputBase(),
        packagePaths,
        eventHandler,
        directories.getWorkspace(),
        workingDirectory,
        buildFilesByPriority);
  }

  private CyclesReporter createCyclesReporter() {
    return new CyclesReporter(
        new TransitiveTargetCycleReporter(packageManager),
        new ActionArtifactCycleReporter(packageManager),
        new ConfiguredTargetCycleReporter(packageManager),
        new TestExpansionCycleReporter(packageManager),
        new RegisteredToolchainsCycleReporter(),
        // TODO(ulfjack): The SkylarkModuleCycleReporter swallows previously reported cycles
        // unconditionally! Is that intentional?
        new SkylarkModuleCycleReporter());
  }

  CyclesReporter getCyclesReporter() {
    return cyclesReporter.get();
  }

  /** Convenience method with same semantics as {@link CyclesReporter#reportCycles}. */
  public void reportCycles(
      ExtendedEventHandler eventHandler, Iterable<CycleInfo> cycles, SkyKey topLevelKey) {
    getCyclesReporter().reportCycles(cycles, topLevelKey, eventHandler);
  }

  public void setActionExecutionProgressReportingObjects(
      @Nullable ProgressSupplier supplier,
      @Nullable ActionCompletedReceiver completionReceiver,
      @Nullable ActionExecutionStatusReporter statusReporter) {
    skyframeActionExecutor.setActionExecutionProgressReportingObjects(supplier, completionReceiver);
    this.statusReporterRef.set(statusReporter);
  }

  public abstract void detectModifiedOutputFiles(
      ModifiedFileSet modifiedOutputFiles, @Nullable Range<Long> lastExecutionTimeRange)
      throws AbruptExitException, InterruptedException;

  /**
   * Mark dirty values for deletion if they've been dirty for longer than N versions.
   *
   * <p>Specifying a value N means, if the current version is V and a value was dirtied (and has
   * remained so) in version U, and U + N &lt;= V, then the value will be marked for deletion and
   * purged in version V+1.
   */
  public abstract void deleteOldNodes(long versionWindowForDirtyGc);

  @Nullable
  public PackageProgressReceiver getPackageProgressReceiver() {
    return packageProgress;
  }

  /**
   * Loads the given target patterns without applying any filters (such as removing non-test targets
   * if {@code --build_tests_only} is set).
   *
   * @param eventHandler handler which accepts update events
   * @param targetPatterns patterns to be loaded
   * @param threadCount number of threads to use for this skyframe evaluation
   * @param keepGoing whether to attempt to ignore errors. See also {@link KeepGoingOption}
   */
  public TargetPatternPhaseValue loadTargetPatternsWithoutFilters(
      ExtendedEventHandler eventHandler,
      List<String> targetPatterns,
      PathFragment relativeWorkingDirectory,
      int threadCount,
      boolean keepGoing)
      throws TargetParsingException, InterruptedException {
    SkyKey key =
        TargetPatternPhaseValue.keyWithoutFilters(
            ImmutableList.copyOf(targetPatterns), relativeWorkingDirectory.getPathString());
    return getTargetPatternPhaseValue(eventHandler, targetPatterns, threadCount, keepGoing, key);
  }

  /**
   * Loads the given target patterns after applying filters configured through parameters and
   * options (such as removing non-test targets if {@code --build_tests_only} is set).
   *
   * @param eventHandler handler which accepts update events
   * @param targetPatterns patterns to be loaded
   * @param threadCount number of threads to use for this skyframe evaluation
   * @param keepGoing whether to attempt to ignore errors. See also {@link KeepGoingOption}
   * @param determineTests whether to ignore any targets that aren't tests or test suites
   */
  public TargetPatternPhaseValue loadTargetPatternsWithFilters(
      ExtendedEventHandler eventHandler,
      List<String> targetPatterns,
      PathFragment relativeWorkingDirectory,
      LoadingOptions options,
      int threadCount,
      boolean keepGoing,
      boolean determineTests)
      throws TargetParsingException, InterruptedException {
    SkyKey key =
        TargetPatternPhaseValue.key(
            ImmutableList.copyOf(targetPatterns),
            relativeWorkingDirectory.getPathString(),
            options.compileOneDependency,
            options.buildTestsOnly,
            determineTests,
            ImmutableList.copyOf(options.buildTagFilterList),
            options.buildManualTests,
            options.expandTestSuites,
            TestFilter.forOptions(options, eventHandler, pkgFactory.getRuleClassNames()));
    return getTargetPatternPhaseValue(eventHandler, targetPatterns, threadCount, keepGoing, key);
  }

  private TargetPatternPhaseValue getTargetPatternPhaseValue(
      ExtendedEventHandler eventHandler,
      List<String> targetPatterns,
      int threadCount,
      boolean keepGoing,
      SkyKey key)
      throws InterruptedException, TargetParsingException {
    Stopwatch timer = Stopwatch.createStarted();
    eventHandler.post(new LoadingPhaseStartedEvent(packageProgress));
    EvaluationResult<TargetPatternPhaseValue> evalResult =
        evaluate(ImmutableList.of(key), keepGoing, threadCount, eventHandler);
    if (evalResult.hasError()) {
      ErrorInfo errorInfo = evalResult.getError(key);
      TargetParsingException exc;
      if (!Iterables.isEmpty(errorInfo.getCycleInfo())) {
        exc = new TargetParsingException("cycles detected during target parsing");
        getCyclesReporter().reportCycles(errorInfo.getCycleInfo(), key, eventHandler);
        // Fallback: we don't know which patterns failed, specifically, so we report the entire
        // set as being in error.
        eventHandler.post(PatternExpandingError.failed(targetPatterns, exc.getMessage()));
      } else {
        Exception e = Preconditions.checkNotNull(errorInfo.getException());

        // TargetPatternPhaseFunction never directly throws. Thus, the only way
        // evalResult.hasError() && keepGoing can hold is if there are cycles, which is handled
        // above.
        Preconditions.checkState(!keepGoing, e.getMessage());

        // Following SkyframeTargetPatternEvaluator, we convert any exception into a
        // TargetParsingException.
        exc =
            (e instanceof TargetParsingException)
                ? (TargetParsingException) e
                : new TargetParsingException(e.getMessage(), e);
        if (!(e instanceof TargetParsingException)) {
          // If it's a TargetParsingException, then the TargetPatternPhaseFunction has already
          // reported the error, so we don't need to report it again.
          eventHandler.post(PatternExpandingError.failed(targetPatterns, exc.getMessage()));
        }
      }
      throw exc;
    }
    eventHandler.post(new TargetParsingPhaseTimeEvent(timer.stop().elapsed().toMillis()));
    return evalResult.get(key);
  }

  public PrepareAnalysisPhaseValue prepareAnalysisPhase(
      ExtendedEventHandler eventHandler,
      BuildOptions buildOptions,
      Set<String> multiCpu,
      Collection<Label> labels)
      throws InvalidConfigurationException, InterruptedException {
    FragmentClassSet allFragments =
        FragmentClassSet.of(
            ruleClassProvider.getConfigurationFragments().stream()
                .map(factory -> factory.creates())
                .collect(
                    ImmutableSortedSet.toImmutableSortedSet(
                        BuildConfiguration.lexicalFragmentSorter)));
    SkyKey key =
        PrepareAnalysisPhaseValue.key(
            allFragments,
            BuildOptions.diffForReconstruction(defaultBuildOptions, buildOptions),
            multiCpu,
            labels);
    EvaluationResult<PrepareAnalysisPhaseValue> evalResult =
        evaluate(
            ImmutableList.of(key),
            /*keepGoing=*/ true,
            /*numThreads=*/ DEFAULT_THREAD_COUNT,
            eventHandler);
    if (evalResult.hasError()) {
      ErrorInfo errorInfo = evalResult.getError(key);
      Exception e = errorInfo.getException();
      if (e == null && !Iterables.isEmpty(errorInfo.getCycleInfo())) {
        getCyclesReporter().reportCycles(errorInfo.getCycleInfo(), key, eventHandler);
        e =
            new InvalidConfigurationException(
                "cannot load build configuration because of this cycle");
      } else if (e instanceof NoSuchThingException) {
        e = new InvalidConfigurationException(e);
      }
      if (e != null) {
        Throwables.throwIfInstanceOf(e, InvalidConfigurationException.class);
      }
      throw new IllegalStateException("Unknown error during configuration creation evaluation", e);
    }

    if (configuredTargetProgress != null) {
      configuredTargetProgress.reset();
    }

    PrepareAnalysisPhaseValue prepareAnalysisPhaseValue = evalResult.get(key);
    return prepareAnalysisPhaseValue;
  }

  /** A progress received to track analysis invalidation and update progress messages. */
  protected class SkyframeProgressReceiver
      extends EvaluationProgressReceiver.NullEvaluationProgressReceiver {
    /**
     * This flag is needed in order to avoid invalidating legacy data when we clear the analysis
     * cache because of --discard_analysis_cache flag. For that case we want to keep the legacy data
     * but get rid of the Skyframe data.
     */
    protected boolean ignoreInvalidations = false;
    /** This receiver is only needed for execution, so it is null otherwise. */
    @Nullable EvaluationProgressReceiver executionProgressReceiver = null;
    /** This receiver is only needed for loading, so it is null otherwise. */
    @Override
    public void invalidated(SkyKey skyKey, InvalidationState state) {
      trimmingListener.invalidated(skyKey, state);
      if (ignoreInvalidations) {
        return;
      }
      skyframeBuildView.getProgressReceiver().invalidated(skyKey, state);
    }

    @Override
    public void enqueueing(SkyKey skyKey) {
      trimmingListener.enqueueing(skyKey);
      if (ignoreInvalidations) {
        return;
      }
      skyframeBuildView.getProgressReceiver().enqueueing(skyKey);
      if (executionProgressReceiver != null) {
        executionProgressReceiver.enqueueing(skyKey);
      }
    }

    @Override
    public void evaluated(
        SkyKey skyKey,
        @Nullable SkyValue value,
        Supplier<EvaluationSuccessState> evaluationSuccessState,
        EvaluationState state) {
      trimmingListener.evaluated(skyKey, value, evaluationSuccessState, state);
      if (ignoreInvalidations) {
        return;
      }
      skyframeBuildView
          .getProgressReceiver()
          .evaluated(skyKey, value, evaluationSuccessState, state);
      if (executionProgressReceiver != null) {
        executionProgressReceiver.evaluated(skyKey, value, evaluationSuccessState, state);
      }
    }
  }

  public abstract ExecutionFinishedEvent createExecutionFinishedEvent();

  protected Iterable<ActionLookupValue> getActionLookupValuesInBuild(
      List<ConfiguredTargetKey> topLevelCtKeys, List<AspectValueKey> aspectKeys)
      throws InterruptedException {
    if (!tracksStateForIncrementality()) {
      // If we do not track incremental state we do not have graph edges, so we cannot traverse the
      // graph and find only actions in the current build. In this case we can simply return all
      // ActionLookupValues in the graph, since the graph's lifetime is a single build anyway.
      return Iterables.filter(memoizingEvaluator.getDoneValues().values(), ActionLookupValue.class);
    }
    WalkableGraph walkableGraph = SkyframeExecutorWrappingWalkableGraph.of(this);
    Set<SkyKey> seen = CompactHashSet.create();
    List<ActionLookupValue> result = new ArrayList<>();
    for (ConfiguredTargetKey key : topLevelCtKeys) {
      findActionsRecursively(walkableGraph, key, seen, result);
    }
    for (AspectValueKey key : aspectKeys) {
      findActionsRecursively(walkableGraph, key, seen, result);
    }
    return result;
  }

  private static void findActionsRecursively(
      WalkableGraph walkableGraph, SkyKey key, Set<SkyKey> seen, List<ActionLookupValue> result)
      throws InterruptedException {
    if (!(key instanceof ActionLookupValue.ActionLookupKey) || !seen.add(key)) {
      // The subgraph of dependencies of ActionLookupValues never has a non-ActionLookupValue
      // depending on an ActionLookupValue. So we can skip any non-ActionLookupValues in the
      // traversal as an optimization.
      return;
    }
    SkyValue value = walkableGraph.getValue(key);
    if (value == null) {
      return; // The value failed to evaluate.
    }
    if (value instanceof ActionLookupValue) {
      result.add((ActionLookupValue) value);
    }
    for (SkyKey dep : walkableGraph.getDirectDeps(key)) {
      findActionsRecursively(walkableGraph, dep, seen, result);
    }
  }

  private <T extends SkyValue> EvaluationResult<T> evaluate(
      Iterable<? extends SkyKey> roots,
      boolean keepGoing,
      int numThreads,
      ExtendedEventHandler eventHandler)
      throws InterruptedException {
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(keepGoing)
            .setNumThreads(numThreads)
            .setEventHander(eventHandler)
            .build();
    return buildDriver.evaluate(roots, evaluationContext);
  }
}
