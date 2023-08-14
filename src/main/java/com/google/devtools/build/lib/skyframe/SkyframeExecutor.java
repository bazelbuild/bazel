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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.concurrent.Uninterruptibles.callUninterruptibly;
import static com.google.devtools.build.lib.skyframe.ArtifactConflictFinder.ACTION_CONFLICTS;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Stopwatch;
import com.google.common.base.Throwables;
import com.google.common.collect.ClassToInstanceMap;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionExecutionStatusReporter;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLogBufferPathGenerator;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CompletionContext.PathResolverFactory;
import com.google.devtools.build.lib.actions.DiscoveredModulesPruner;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.MapBasedActionGraph;
import com.google.devtools.build.lib.actions.PackageRoots;
import com.google.devtools.build.lib.actions.RemoteArtifactChecker;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.AnalysisOptions;
import com.google.devtools.build.lib.analysis.AspectConfiguredEvent;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.InconsistentNullConfigException;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.TargetConfiguredEvent;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TransitiveDependencyState;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.Factory;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.constraints.RuleContextConstraintSemantics;
import com.google.devtools.build.lib.analysis.producers.ConfiguredTargetAndDataProducer;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleValue;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.Label.LabelInterner;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.collect.nestedset.ArtifactNestedSetKey;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.concurrent.NamedForkJoinPool;
import com.google.devtools.build.lib.concurrent.QuiescingExecutor;
import com.google.devtools.build.lib.concurrent.QuiescingExecutors;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.ErrorSensingEventHandler;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.io.FileSymlinkCycleUniquenessFunction;
import com.google.devtools.build.lib.io.FileSymlinkInfiniteExpansionUniquenessFunction;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.Builder.PackageSettings;
import com.google.devtools.build.lib.packages.Package.ConfigSettingVisibilityPolicy;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.TargetParsingPhaseTimeEvent;
import com.google.devtools.build.lib.pkgcache.TargetPatternPreloader;
import com.google.devtools.build.lib.pkgcache.TestFilter;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.common.QueryTransitivePackagePreloader;
import com.google.devtools.build.lib.query2.common.UniverseScope;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.repository.ExternalPackageHelper;
import com.google.devtools.build.lib.rules.genquery.GenQueryConfiguration.GenQueryOptions;
import com.google.devtools.build.lib.rules.genquery.GenQueryDirectPackageProviderFactory;
import com.google.devtools.build.lib.rules.repository.ResolvedFileFunction;
import com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.BuildConfiguration.Code;
import com.google.devtools.build.lib.server.FailureDetails.ExternalRepository;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.TargetPatterns;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue.ActionTemplateExpansionKey;
import com.google.devtools.build.lib.skyframe.ArtifactConflictFinder.ConflictException;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.TopLevelAspectsKey;
import com.google.devtools.build.lib.skyframe.BuildDriverFunction.ActionLookupValuesCollectionResult;
import com.google.devtools.build.lib.skyframe.BuildDriverFunction.TestTypeResolver;
import com.google.devtools.build.lib.skyframe.BuildDriverFunction.TransitiveActionLookupValuesHelper;
import com.google.devtools.build.lib.skyframe.DiffAwarenessManager.ProcessableModifiedFileSet;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.ExternalDirtinessChecker;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.FileDirtinessChecker;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.MissingDiffDirtinessChecker;
import com.google.devtools.build.lib.skyframe.DirtinessCheckerUtils.UnionDirtinessChecker;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFilesKnowledge;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.FileType;
import com.google.devtools.build.lib.skyframe.FilesystemValueChecker.ImmutableBatchDirtyResult;
import com.google.devtools.build.lib.skyframe.MetadataConsumerForMetrics.FilesMetricConsumer;
import com.google.devtools.build.lib.skyframe.PackageFunction.ActionOnIOExceptionReadingBuildFile;
import com.google.devtools.build.lib.skyframe.PackageFunction.GlobbingStrategy;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue.RepositoryMappingResolutionException;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ActionCompletedReceiver;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ProgressSupplier;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.SomeExecutionStartedEvent;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredExecutionPlatformsFunction;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredToolchainsCycleReporter;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredToolchainsFunction;
import com.google.devtools.build.lib.skyframe.toolchains.SingleToolchainResolutionFunction;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainResolutionFunction;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.util.ResourceUsage;
import com.google.devtools.build.lib.util.TestType;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.Differencer;
import com.google.devtools.build.skyframe.Differencer.DiffWithDelta.Delta;
import com.google.devtools.build.skyframe.EmittedEventState;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationContext.UnnecessaryTemporaryStateDropper;
import com.google.devtools.build.skyframe.EvaluationContext.UnnecessaryTemporaryStateDropperReceiver;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.EventFilter;
import com.google.devtools.build.skyframe.GroupedDeps;
import com.google.devtools.build.skyframe.ImmutableDiff;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.Injectable;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.NodeEntry;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.devtools.build.skyframe.WalkableGraph.WalkableGraphFactory;
import com.google.devtools.build.skyframe.state.StateMachineEvaluatorForTesting;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;
import com.google.devtools.common.options.ParsedOptionDescription;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.errorprone.annotations.ForOverride;
import java.io.PrintStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Lock;
import java.util.function.Consumer;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * A helper object to support Skyframe-driven execution.
 *
 * <p>This object is mostly used to inject external state, such as the executor engine or some
 * additional artifacts (workspace status and build info artifacts) into SkyFunctions for use during
 * the build.
 */
public abstract class SkyframeExecutor implements WalkableGraphFactory {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  protected MemoizingEvaluator memoizingEvaluator;
  protected final EmittedEventState emittedEventState = new EmittedEventState();
  protected final PackageFactory pkgFactory;
  private final WorkspaceStatusAction.Factory workspaceStatusActionFactory;
  protected final FileSystem fileSystem;
  protected final BlazeDirectories directories;
  final ExternalFilesHelper externalFilesHelper;
  protected final BugReporter bugReporter;

  /**
   * Measures source artifacts read this build. Does not include cached artifacts, so is less useful
   * on incremental builds.
   */
  private final FilesMetricConsumer sourceArtifactsSeen = new FilesMetricConsumer();

  private final FilesMetricConsumer outputArtifactsSeen = new FilesMetricConsumer();
  private final FilesMetricConsumer outputArtifactsFromActionCache = new FilesMetricConsumer();
  private final FilesMetricConsumer topLevelArtifactsMetric = new FilesMetricConsumer();

  @Nullable OutputService outputService;

  // TODO(bazel-team): Figure out how to handle value builders that block internally. Blocking
  // operations may need to be handled in another (bigger?) thread pool. Also, we should detect
  // the number of cores and use that as the thread-pool size for CPU-bound operations.
  // I just bumped this to 200 to get reasonable execution phase performance; that may cause
  // significant overhead for CPU-bound processes (i.e. analysis). [skyframe-analysis]
  public static final int DEFAULT_THREAD_COUNT =
      // Reduce thread count while running tests of Bazel. Test cases are typically small, and large
      // thread pools vying for a relatively small number of CPU cores may induce non-optimal
      // performance.
      TestType.isInTest() ? 5 : 200;

  // The limit of how many times we will traverse through an exception chain when catching a
  // target parsing exception.
  private static final int EXCEPTION_TRAVERSAL_LIMIT = 10;

  // Cache of parsed bzl files, for use when we're inlining BzlCompileFunction in
  // BzlLoadFunction. See the comments in BzlLoadFunction for motivations and details.
  private final Cache<BzlCompileValue.Key, BzlCompileValue> bzlCompileCache =
      Caffeine.newBuilder().build();

  private final AtomicInteger numPackagesSuccessfullyLoaded = new AtomicInteger(0);
  @Nullable private final PackageProgressReceiver packageProgress;
  @Nullable private final ConfiguredTargetProgressReceiver configuredTargetProgress;
  protected final SyscallCache syscallCache;

  protected final SkyframeBuildView skyframeBuildView;
  private ActionLogBufferPathGenerator actionLogBufferPathGenerator;

  private final Consumer<SkyframeExecutor> skyframeExecutorConsumerOnInit;

  // AtomicReferences are used here as mutable boxes shared with value builders.
  private final AtomicBoolean showLoadingProgress = new AtomicBoolean();
  private final AtomicReference<PathPackageLocator> pkgLocator = new AtomicReference<>();
  final AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages =
      new AtomicReference<>(ImmutableSet.of());
  private final AtomicReference<EventBus> eventBus = new AtomicReference<>();
  final AtomicReference<TimestampGranularityMonitor> tsgm = new AtomicReference<>();
  private final AtomicReference<Map<String, String>> clientEnv = new AtomicReference<>();

  private final ArtifactFactory artifactFactory;
  private final ActionKeyContext actionKeyContext;

  boolean active = true;
  private final SkyframePackageManager packageManager;
  private final QueryTransitivePackagePreloader queryTransitivePackagePreloader;

  /** Used to lock evaluator on legacy calls to get existing values. */
  private final Object valueLookupLock = new Object();

  private final AtomicReference<ActionExecutionStatusReporter> statusReporterRef =
      new AtomicReference<>();
  protected final SkyframeActionExecutor skyframeActionExecutor;
  private ActionExecutionFunction actionExecutionFunction;
  private BuildDriverFunction buildDriverFunction;
  private GlobFunction globFunction;
  SkyframeProgressReceiver progressReceiver;
  private CyclesReporter cyclesReporter = null;

  @VisibleForTesting boolean lastAnalysisDiscarded = false;

  /**
   * True if analysis was not incremental because {@link #handleAnalysisInvalidatingChange} was
   * called, typically because a configuration-related option changed.
   */
  private boolean analysisCacheInvalidated = false;

  /** True if loading and analysis nodes were cleared (discarded) after analysis to save memory. */
  private boolean analysisCacheCleared;

  private final ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions;

  SkyframeIncrementalBuildMonitor incrementalBuildMonitor = new SkyframeIncrementalBuildMonitor();

  private final SkyFunction ignoredPackagePrefixesFunction;

  private final ConfiguredRuleClassProvider ruleClassProvider;

  private final CrossRepositoryLabelViolationStrategy crossRepositoryLabelViolationStrategy;

  private final ImmutableList<BuildFileName> buildFilesByPriority;

  private final ExternalPackageHelper externalPackageHelper;

  private final ActionOnIOExceptionReadingBuildFile actionOnIOExceptionReadingBuildFile;

  private final boolean shouldUseRepoDotBazel;

  private final boolean shouldUnblockCpuWorkWhenFetchingDeps;

  private final SkyKeyStateReceiver skyKeyStateReceiver;

  private final PathResolverFactory pathResolverFactory = new PathResolverFactoryImpl();

  // A Semaphore to limit the number of in-flight execution of certain SkyFunctions to prevent OOM.
  // TODO(b/185987566): Remove this semaphore.
  private static final int DEFAULT_SEMAPHORE_SIZE = ResourceUsage.getAvailableProcessors();
  private final AtomicReference<Semaphore> oomSensitiveSkyFunctionsSemaphore =
      new AtomicReference<>(new Semaphore(DEFAULT_SEMAPHORE_SIZE));

  private Map<String, String> lastRemoteDefaultExecProperties;
  private RemoteOutputsMode lastRemoteOutputsMode;
  private Boolean lastRemoteCacheEnabled;

  // start: Skymeld-only
  // This is set once every build and set to null at the end of each.
  @Nullable private Supplier<Boolean> mergedSkyframeAnalysisExecutionSupplier;

  // Reset after each build.
  @Nullable private IncrementalArtifactConflictFinder incrementalArtifactConflictFinder;
  // Reset after each build.
  @Nullable
  private TransitiveActionLookupKeysCollector incrementalTransitiveActionLookupKeysCollector;
  // A set of ActionLookupKeys which have been confirmed to be conflict-free. This is used for
  // pruning while going through the ActionLookupKeys in the build for conflict checking.
  // Reset after each build.
  private Set<ActionLookupKey> conflictFreeActionLookupKeysGlobalSet;
  // end: Skymeld-only

  private RuleContextConstraintSemantics ruleContextConstraintSemantics;
  private RegexFilter extraActionFilter;
  @Nullable private ActionExecutionInactivityWatchdog watchdog;

  private final AtomicBoolean isBuildingExclusiveArtifacts = new AtomicBoolean(false);

  // Reset to null after each build to save memory. Guaranteed to be non-null when retrieved via
  // BuildDriverFunction.
  @Nullable private TestTypeResolver testTypeResolver;

  // This boolean controls whether FILE_STATE or DIRECTORY_LISTING_STATE nodes are dropped after the
  // corresponding FILE or DIRECTORY_LISTING nodes are evaluated.
  // See b/261019506.
  protected boolean heuristicallyDropNodes = false;

  final AtomicInteger modifiedFiles = new AtomicInteger();
  int numSourceFilesCheckedBecauseOfMissingDiffs;
  // This is intentionally not kept in sync with the evaluator: we may reset the evaluator without
  // ever losing injected/invalidated data here. This is safe because the worst that will happen is
  // that on the next build we try to inject/invalidate some nodes that aren't needed for the build.
  @Nullable protected final RecordingDifferencer recordingDiffer;
  @Nullable protected final DiffAwarenessManager diffAwarenessManager;
  // If this is null then workspace header pre-calculation won't happen.
  @Nullable private final SkyframeExecutorRepositoryHelpersHolder repositoryHelpersHolder;
  private boolean repositoryHelpersHolderIgnored = false;
  @Nullable private final WorkspaceInfoFromDiffReceiver workspaceInfoFromDiffReceiver;
  private Set<String> previousClientEnvironment = ImmutableSet.of();

  protected Duration sourceDiffCheckingDuration = Duration.ofSeconds(-1L);

  final class PathResolverFactoryImpl implements PathResolverFactory {
    @Override
    public boolean shouldCreatePathResolverForArtifactValues() {
      return outputService != null && outputService.supportsPathResolverForArtifactValues();
    }

    @Override
    public ArtifactPathResolver createPathResolverForArtifactValues(
        ActionInputMap actionInputMap,
        Map<Artifact, ImmutableCollection<? extends Artifact>> expandedArtifacts,
        Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesets,
        String workspaceName) {
      checkState(shouldCreatePathResolverForArtifactValues());
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
      PackageFactory pkgFactory,
      FileSystem fileSystem,
      BlazeDirectories directories,
      ActionKeyContext actionKeyContext,
      Factory workspaceStatusActionFactory,
      ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions,
      SyscallCache syscallCache,
      ExternalFileAction externalFileAction,
      SkyFunction ignoredPackagePrefixesFunction,
      CrossRepositoryLabelViolationStrategy crossRepositoryLabelViolationStrategy,
      ImmutableList<BuildFileName> buildFilesByPriority,
      ExternalPackageHelper externalPackageHelper,
      ActionOnIOExceptionReadingBuildFile actionOnIOExceptionReadingBuildFile,
      boolean shouldUseRepoDotBazel,
      boolean shouldUnblockCpuWorkWhenFetchingDeps,
      @Nullable PackageProgressReceiver packageProgress,
      @Nullable ConfiguredTargetProgressReceiver configuredTargetProgress,
      SkyKeyStateReceiver skyKeyStateReceiver,
      BugReporter bugReporter,
      @Nullable Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories,
      @Nullable WorkspaceInfoFromDiffReceiver workspaceInfoFromDiffReceiver,
      @Nullable RecordingDifferencer recordingDiffer,
      @Nullable SkyframeExecutorRepositoryHelpersHolder repositoryHelpersHolder) {
    // Strictly speaking, these arguments are not required for initialization, but all current
    // callsites have them at hand, so we might as well set them during construction.
    this.skyframeExecutorConsumerOnInit = skyframeExecutorConsumerOnInit;
    this.pkgFactory = pkgFactory;
    this.shouldUnblockCpuWorkWhenFetchingDeps = shouldUnblockCpuWorkWhenFetchingDeps;
    this.skyKeyStateReceiver = skyKeyStateReceiver;
    this.bugReporter = bugReporter;
    this.syscallCache = syscallCache;
    this.pkgFactory.setSyscallCache(this.syscallCache);
    this.workspaceStatusActionFactory = workspaceStatusActionFactory;
    this.queryTransitivePackagePreloader =
        new QueryTransitivePackagePreloader(
            () -> memoizingEvaluator, this::newEvaluationContextBuilder, bugReporter);
    this.packageManager =
        new SkyframePackageManager(
            new SkyframePackageLoader(),
            this.syscallCache,
            pkgLocator::get,
            numPackagesSuccessfullyLoaded);
    this.fileSystem = fileSystem;
    this.directories = checkNotNull(directories);
    this.actionKeyContext = checkNotNull(actionKeyContext);
    this.ignoredPackagePrefixesFunction = ignoredPackagePrefixesFunction;
    this.extraSkyFunctions = extraSkyFunctions;

    this.ruleClassProvider = (ConfiguredRuleClassProvider) pkgFactory.getRuleClassProvider();
    this.skyframeActionExecutor =
        new SkyframeActionExecutor(
            actionKeyContext,
            outputArtifactsSeen,
            outputArtifactsFromActionCache,
            statusReporterRef,
            this::getPathEntries,
            this.syscallCache,
            skyKeyStateReceiver::makeThreadStateReceiver);
    this.artifactFactory =
        new ArtifactFactory(
            /*execRootParent=*/ directories.getExecRootBase(), directories.getRelativeOutputPath());
    this.skyframeBuildView =
        new SkyframeBuildView(artifactFactory, this, ruleClassProvider, actionKeyContext);
    this.externalFilesHelper =
        ExternalFilesHelper.create(pkgLocator, externalFileAction, directories);
    this.crossRepositoryLabelViolationStrategy = crossRepositoryLabelViolationStrategy;
    this.buildFilesByPriority = buildFilesByPriority;
    this.externalPackageHelper = externalPackageHelper;
    this.actionOnIOExceptionReadingBuildFile = actionOnIOExceptionReadingBuildFile;
    this.shouldUseRepoDotBazel = shouldUseRepoDotBazel;
    this.packageProgress = packageProgress;
    this.configuredTargetProgress = configuredTargetProgress;
    this.diffAwarenessManager =
        diffAwarenessFactories != null ? new DiffAwarenessManager(diffAwarenessFactories) : null;
    this.workspaceInfoFromDiffReceiver = workspaceInfoFromDiffReceiver;
    this.recordingDiffer = recordingDiffer;
    this.repositoryHelpersHolder = repositoryHelpersHolder;
  }

  private ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions() {
    BzlLoadFunction bzlLoadFunctionForInliningPackageAndWorkspaceNodes =
        getBzlLoadFunctionForInliningPackageAndWorkspaceNodes();

    // We don't check for duplicates in order to allow extraSkyfunctions to override existing
    // entries.
    Map<SkyFunctionName, SkyFunction> map = new HashMap<>();
    // IF YOU ADD A NEW SKYFUNCTION: If your Skyfunction can be used transitively by package
    // loading, make sure to register it in AbstractPackageLoader as well.
    map.put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction());
    map.put(SkyFunctions.CLIENT_ENVIRONMENT_VARIABLE, new ClientEnvironmentFunction(clientEnv));
    map.put(SkyFunctions.ACTION_ENVIRONMENT_VARIABLE, new ActionEnvironmentFunction());
    map.put(FileStateKey.FILE_STATE, newFileStateFunction());
    map.put(SkyFunctions.DIRECTORY_LISTING_STATE, newDirectoryListingStateFunction());
    map.put(FileSymlinkCycleUniquenessFunction.NAME, new FileSymlinkCycleUniquenessFunction());
    map.put(
        FileSymlinkInfiniteExpansionUniquenessFunction.NAME,
        new FileSymlinkInfiniteExpansionUniquenessFunction());
    map.put(FileValue.FILE, new FileFunction(pkgLocator, directories));
    map.put(SkyFunctions.DIRECTORY_LISTING, new DirectoryListingFunction());
    map.put(
        SkyFunctions.PACKAGE_LOOKUP,
        new PackageLookupFunction(
            deletedPackages,
            crossRepositoryLabelViolationStrategy,
            buildFilesByPriority,
            externalPackageHelper));
    map.put(SkyFunctions.CONTAINING_PACKAGE_LOOKUP, new ContainingPackageLookupFunction());
    map.put(
        SkyFunctions.BZL_COMPILE, // TODO rename
        new BzlCompileFunction(
            ruleClassProvider.getBazelStarlarkEnvironment(),
            getDigestFunction().getHashFunction()));
    map.put(
        SkyFunctions.STARLARK_BUILTINS,
        new StarlarkBuiltinsFunction(ruleClassProvider.getBazelStarlarkEnvironment()));
    map.put(SkyFunctions.BZL_LOAD, newBzlLoadFunction(ruleClassProvider));
    GlobFunction globFunction = newGlobFunction();
    map.put(SkyFunctions.GLOB, globFunction);
    this.globFunction = globFunction;
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
    map.put(SkyFunctions.IGNORED_PACKAGE_PREFIXES, ignoredPackagePrefixesFunction);
    map.put(SkyFunctions.TESTS_IN_SUITE, new TestExpansionFunction());
    map.put(SkyFunctions.TEST_SUITE_EXPANSION, new TestsForTargetPatternFunction());
    map.put(SkyFunctions.TARGET_PATTERN_PHASE, new TargetPatternPhaseFunction());
    map.put(SkyFunctions.RECURSIVE_PKG, new RecursivePkgFunction(directories));
    map.put(
        SkyFunctions.PACKAGE,
        new PackageFunction(
            pkgFactory,
            packageManager,
            showLoadingProgress,
            numPackagesSuccessfullyLoaded,
            bzlLoadFunctionForInliningPackageAndWorkspaceNodes,
            packageProgress,
            actionOnIOExceptionReadingBuildFile,
            shouldUseRepoDotBazel,
            getGlobbingStrategy(),
            skyKeyStateReceiver::makeThreadStateReceiver));
    map.put(SkyFunctions.PACKAGE_ERROR, new PackageErrorFunction());
    map.put(SkyFunctions.PACKAGE_ERROR_MESSAGE, new PackageErrorMessageFunction());
    map.put(SkyFunctions.TARGET_PATTERN_ERROR, new TargetPatternErrorFunction());
    map.put(TransitiveTargetKey.NAME, new TransitiveTargetFunction());
    map.put(Label.TRANSITIVE_TRAVERSAL, getTransitiveTraversalFunction());
    map.put(
        SkyFunctions.CONFIGURED_TARGET,
        new ConfiguredTargetFunction(
            new BuildViewProvider(),
            ruleClassProvider,
            oomSensitiveSkyFunctionsSemaphore,
            shouldStoreTransitivePackagesInLoadingAndAnalysis(),
            shouldUnblockCpuWorkWhenFetchingDeps,
            configuredTargetProgress,
            this::getExistingPackage));
    map.put(
        SkyFunctions.ASPECT,
        new AspectFunction(
            new BuildViewProvider(),
            shouldStoreTransitivePackagesInLoadingAndAnalysis(),
            this::getExistingPackage));
    map.put(SkyFunctions.TOP_LEVEL_ASPECTS, new ToplevelStarlarkAspectFunction());
    map.put(
        SkyFunctions.BUILD_TOP_LEVEL_ASPECTS_DETAILS, new BuildTopLevelAspectsDetailsFunction());
    map.put(
        GenQueryDirectPackageProviderFactory.GENQUERY_SCOPE,
        GenQueryDirectPackageProviderFactory.FUNCTION);
    map.put(SkyFunctions.ACTION_LOOKUP_CONFLICT_FINDING, new ActionLookupConflictFindingFunction());
    map.put(
        SkyFunctions.TOP_LEVEL_ACTION_LOOKUP_CONFLICT_FINDING,
        new TopLevelActionLookupConflictFindingFunction());
    map.put(
        SkyFunctions.BUILD_CONFIGURATION,
        new BuildConfigurationFunction(directories, ruleClassProvider));
    map.put(SkyFunctions.BASELINE_OPTIONS, new BaselineOptionsFunction());
    map.put(
        SkyFunctions.STARLARK_BUILD_SETTINGS_DETAILS, new StarlarkBuildSettingsDetailsFunction());
    map.put(SkyFunctions.WORKSPACE_NAME, new WorkspaceNameFunction());
    map.put(
        WorkspaceFileValue.WORKSPACE_FILE,
        new WorkspaceFileFunction(
            ruleClassProvider,
            pkgFactory,
            directories,
            bzlLoadFunctionForInliningPackageAndWorkspaceNodes));
    map.put(
        SkyFunctions.REPO_FILE,
        shouldUseRepoDotBazel
            ? new RepoFileFunction(
            ruleClassProvider.getBazelStarlarkEnvironment(), directories.getWorkspace())
            : (k, env) -> {
              throw new IllegalStateException("supposed to be unused");
            });
    map.put(SkyFunctions.EXTERNAL_PACKAGE, new ExternalPackageFunction(externalPackageHelper));
    map.put(
        BzlmodRepoRuleValue.BZLMOD_REPO_RULE,
        new BzlmodRepoRuleFunction(ruleClassProvider, directories));
    map.put(
        SkyFunctions.TARGET_COMPLETION,
        TargetCompletor.targetCompletionFunction(
            pathResolverFactory, skyframeActionExecutor, topLevelArtifactsMetric, bugReporter));
    map.put(
        SkyFunctions.ASPECT_COMPLETION,
        AspectCompletor.aspectCompletionFunction(
            pathResolverFactory, skyframeActionExecutor, topLevelArtifactsMetric, bugReporter));
    map.put(SkyFunctions.TEST_COMPLETION, new TestCompletionFunction());
    map.put(
        Artifact.ARTIFACT,
        new ArtifactFunction(
            () -> !skyframeActionExecutor.actionFileSystemType().inMemoryFileSystem(),
            sourceArtifactsSeen,
            syscallCache));
    map.put(
        SkyFunctions.BUILD_INFO_COLLECTION,
        new BuildInfoCollectionFunction(actionKeyContext, artifactFactory));
    map.put(SkyFunctions.BUILD_INFO, new WorkspaceStatusFunction(this::makeWorkspaceStatusAction));
    map.put(SkyFunctions.COVERAGE_REPORT, new CoverageReportFunction(actionKeyContext));
    ActionExecutionFunction actionExecutionFunction =
        new ActionExecutionFunction(skyframeActionExecutor, directories, tsgm::get, bugReporter);
    map.put(SkyFunctions.ACTION_EXECUTION, actionExecutionFunction);
    this.actionExecutionFunction = actionExecutionFunction;
    map.put(
        SkyFunctions.RECURSIVE_FILESYSTEM_TRAVERSAL,
        new RecursiveFilesystemTraversalFunction(syscallCache));
    map.put(SkyFunctions.FILESET_ENTRY, new FilesetEntryFunction(directories::getExecRoot));
    map.put(
        SkyFunctions.ACTION_TEMPLATE_EXPANSION,
        new ActionTemplateExpansionFunction(actionKeyContext));
    map.put(
        SkyFunctions.LOCAL_REPOSITORY_LOOKUP,
        new LocalRepositoryLookupFunction(externalPackageHelper));
    map.put(
        SkyFunctions.REGISTERED_EXECUTION_PLATFORMS, new RegisteredExecutionPlatformsFunction());
    map.put(SkyFunctions.REGISTERED_TOOLCHAINS, new RegisteredToolchainsFunction());
    map.put(SkyFunctions.SINGLE_TOOLCHAIN_RESOLUTION, new SingleToolchainResolutionFunction());
    map.put(SkyFunctions.TOOLCHAIN_RESOLUTION, new ToolchainResolutionFunction());
    map.put(SkyFunctions.REPOSITORY_MAPPING, new RepositoryMappingFunction());
    map.put(SkyFunctions.RESOLVED_HASH_VALUES, new ResolvedHashesFunction());
    map.put(SkyFunctions.RESOLVED_FILE, new ResolvedFileFunction());
    map.put(
        SkyFunctions.PLATFORM_MAPPING,
        new PlatformMappingFunction(ruleClassProvider.getFragmentRegistry().getOptionsClasses()));
    map.put(SkyFunctions.ARTIFACT_NESTED_SET, ArtifactNestedSetFunction.createInstance());
    BuildDriverFunction buildDriverFunction =
        new BuildDriverFunction(
            new TransitiveActionLookupValuesHelper() {
              @Override
              public ActionLookupValuesCollectionResult collect(ActionLookupKey key) {
                return collectTransitiveActionLookupValuesOfKey(key);
              }

              @Override
              public void registerConflictFreeKeys(ImmutableSet<ActionLookupKey> keys) {
                getConflictFreeActionLookupKeysGlobalSet().addAll(keys);
              }
            },
            this::getIncrementalArtifactConflictFinder,
            this::getRuleContextConstraintSemantics,
            this::getExtraActionFilter,
            this::getTestTypeResolver);
    map.put(SkyFunctions.BUILD_DRIVER, buildDriverFunction);
    this.buildDriverFunction = buildDriverFunction;

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
    return new FileStateFunction(tsgm::get, syscallCache, externalFilesHelper);
  }

  protected SkyFunction newDirectoryListingStateFunction() {
    return new DirectoryListingStateFunction(externalFilesHelper, syscallCache);
  }

  protected GlobFunction newGlobFunction() {
    return new GlobFunction(/*alwaysUseDirListing=*/ false);
  }

  protected SkyFunction newCollectPackagesUnderDirectoryFunction(BlazeDirectories directories) {
    return new CollectPackagesUnderDirectoryFunction(directories);
  }

  @Nullable
  protected BzlLoadFunction getBzlLoadFunctionForInliningPackageAndWorkspaceNodes() {
    return null;
  }

  protected SkyFunction newBzlLoadFunction(RuleClassProvider ruleClassProvider) {
    return BzlLoadFunction.create(
        ruleClassProvider, directories, getDigestFunction().getHashFunction(), bzlCompileCache);
  }

  @ThreadCompatible
  public void setActive(boolean active) {
    this.active = active;
  }

  protected void checkActive() {
    checkState(active);
  }

  public void configureActionExecutor(
      InputMetadataProvider fileCache, ActionInputPrefetcher actionInputPrefetcher) {
    skyframeActionExecutor.configure(
        fileCache, actionInputPrefetcher, DiscoveredModulesPruner.DEFAULT);
  }

  @ForOverride
  protected abstract void dumpPackages(PrintStream out);

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

  /**
   * Was there an analysis-invalidating change, like a configuration option changing, causing a
   * non-incremental analysis phase to be performed. Calling this resets the state to false.
   */
  public final boolean wasAnalysisCacheInvalidatedAndResetBit() {
    boolean tmp = analysisCacheInvalidated;
    analysisCacheInvalidated = false;
    return tmp;
  }

  /** Was the analysis (and loading) cache cleared to save memory before execution. */
  public final boolean wasAnalysisCacheCleared() {
    return analysisCacheCleared;
  }

  /**
   * This method exists only to allow a module to make a top-level Skyframe call during the
   * transition to making it fully Skyframe-compatible. Do not add additional callers!
   */
  public final SkyValue evaluateSkyKeyForExecutionSetup(
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
        return checkNotNull(result.get(key), "%s %s", result, key);
      }
      ErrorInfo errorInfo = checkNotNull(result.getError(key), "%s %s", key, result);
      Throwables.propagateIfPossible(errorInfo.getException(), EnvironmentalExecException.class);
      if (errorInfo.getException() != null) {
        throw new IllegalStateException(errorInfo.getException());
      }
      throw new IllegalStateException(errorInfo.toString());
    }
  }

  final class BuildViewProvider {
    /** Returns the current {@link SkyframeBuildView} instance. */
    SkyframeBuildView getSkyframeBuildView() {
      return skyframeBuildView;
    }
  }

  /**
   * Must be called before the {@link SkyframeExecutor} can be used (should only be called in
   * factory methods and as an implementation detail of {@link #resetEvaluator}).
   */
  protected final void init() {
    progressReceiver = newSkyframeProgressReceiver();
    memoizingEvaluator = createEvaluator(skyFunctions(), progressReceiver, emittedEventState);
    skyframeExecutorConsumerOnInit.accept(this);

  }

  @ForOverride
  protected abstract MemoizingEvaluator createEvaluator(
      ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions,
      SkyframeProgressReceiver progressReceiver,
      EmittedEventState emittedEventState);

  /**
   * Use the fact that analysis of a target must occur before execution of that target, and in a
   * separate Skyframe evaluation, to avoid propagating events from configured target nodes (and
   * more generally action lookup nodes) to action execution nodes. We take advantage of the fact
   * that if a node depends on an action lookup node and is not itself an action lookup node, then
   * it is an execution-phase node: the action lookup nodes are terminal in the analysis phase.
   *
   * <p>Skymeld: propagate events to BuildDriverKey nodes, since they cover both analysis &
   * execution.
   */
  public static final EventFilter DEFAULT_EVENT_FILTER_WITH_ACTIONS =
      new EventFilter() {
        @Override
        public boolean storeEvents() {
          return true;
        }

        @Override
        public boolean shouldPropagate(SkyKey depKey, SkyKey primaryKey) {
          // Do not propagate events from analysis phase nodes to execution phase nodes.
          return isAnalysisPhaseKey(primaryKey)
              || !isAnalysisPhaseKey(depKey)
              // Skymeld only.
              || primaryKey instanceof BuildDriverKey;
        }
      };

  private static boolean isAnalysisPhaseKey(SkyKey key) {
    return key instanceof ActionLookupKey && !(key instanceof ActionTemplateExpansionKey);
  }

  protected SkyframeProgressReceiver newSkyframeProgressReceiver() {
    return new SkyframeProgressReceiver();
  }

  /** Reinitializes the Skyframe evaluator, dropping all previously computed values. */
  public void resetEvaluator() {
    memoizingEvaluator.cleanupInterningPools();
    init();
    emittedEventState.clear();
    skyframeBuildView.reset();
  }

  /**
   * Notifies the executor that the command is complete. May safely be called multiple times for a
   * single command, so callers should err on the side of calling it more frequently. Should be
   * idempotent, so that calls after the first one in the same evaluation should be quick.
   */
  public void notifyCommandComplete(ExtendedEventHandler eventHandler) throws InterruptedException {
    try {
      memoizingEvaluator.noteEvaluationsAtSameVersionMayBeFinished(eventHandler);
    } finally {
      progressReceiver.globDeps = new ConcurrentHashMap<>();
      globFunction.complete();
      clearSyscallCache();
      // So that the supplier object can be GC-ed.
      mergedSkyframeAnalysisExecutionSupplier = null;
    }
  }

  /**
   * Notifies the executor to post logging stats when the server is crashing, so that logging is
   * still available even when the server crashes.
   */
  public void postLoggingStatsWhenCrashing(ExtendedEventHandler eventHandler) {
    memoizingEvaluator.postLoggingStats(eventHandler);
  }

  /** Clear any configured target data stored outside Skyframe. */
  public void handleAnalysisInvalidatingChange() {
    logger.atInfo().log("Dropping configured target data");
    analysisCacheInvalidated = true;
    skyframeBuildView.clearInvalidatedActionLookupKeys();
    skyframeBuildView.clearLegacyData();
    ArtifactNestedSetFunction.getInstance().resetArtifactNestedSetFunctionMaps();
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
  public abstract List<RuleStat> getRuleStats(ExtendedEventHandler eventHandler)
      throws InterruptedException;

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
      boolean heuristicallyDropNodes,
      boolean discardAnalysisCache,
      EventHandler eventHandler) {
    // Assume incrementality.
  }

  /**
   * Whether this executor tracks state for the purpose of improving incremental performance.
   *
   * <p>A return of {@code false} indicates that nodes have a lifetime of a single command and that
   * graph edges are not kept.
   */
  public boolean tracksStateForIncrementality() {
    return true;
  }

  @ForOverride
  protected GlobbingStrategy getGlobbingStrategy() {
    return tracksStateForIncrementality()
        ? GlobbingStrategy.SKYFRAME_HYBRID
        : GlobbingStrategy.NON_SKYFRAME;
  }

  @ForOverride
  protected boolean shouldDeleteActionNodesWhenDroppingAnalysis() {
    return true;
  }

  /**
   * If not null, this is the only source root in the build, corresponding to the single element in
   * a single-element package path. Such a single-source-root build need not plant the execroot
   * symlink forest, and can trivially resolve source artifacts from exec paths. As a consequence,
   * builds where this is not null do not need to track a package -> source root map. In addition,
   * such builds can only occur in a monorepo, and thus do not need to produce repo mapping
   * manifests for runfiles.
   */
  // TODO(wyv): To be safe, fail early if we're in a multi-repo setup but this is not being tracked.
  @Nullable
  public Root getForcedSingleSourceRootIfNoExecrootSymlinkCreation() {
    return null;
  }

  private boolean shouldStoreTransitivePackagesInLoadingAndAnalysis() {
    // Transitive packages may be needed for either RepoMappingManifestAction or Skymeld with
    // external repository support. They are never needed if external repositories are disabled. To
    // avoid complexity from toggling this, just choose a setting for the lifetime of the server.
    // TODO(b/283125139): Can we support external repositories without tracking transitive packages?
    return repositoryHelpersHolder != null && !repositoryHelpersHolderIgnored;
  }

  @VisibleForTesting
  public void ignoreRepositoryHelpersHolderForTesting() {
    this.repositoryHelpersHolderIgnored = true;
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
          SkyFunctions.PACKAGE, SkyFunctions.BZL_LOAD, SkyFunctions.BZL_COMPILE, SkyFunctions.GLOB);

  /** Data that should be discarded in {@link #discardPreExecutionCache}. */
  protected enum DiscardType {
    ALL,
    ANALYSIS_REFS_ONLY,
    LOADING_NODES_ONLY;

    public boolean discardsAnalysis() {
      return this != LOADING_NODES_ONLY;
    }

    public boolean discardsLoading() {
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
      ImmutableSet<ConfiguredTarget> topLevelTargets,
      ImmutableSet<AspectKey> topLevelAspects,
      DiscardType discardType) {
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
    try (SilentCloseable p = trackDiscardAnalysisCache(discardType)) {
      lastAnalysisDiscarded = true;
      InMemoryGraph graph = memoizingEvaluator.getInMemoryGraph();
      graph.parallelForEach(
          e -> {
            if (!e.isDone()) {
              return;
            }
            boolean removeNode =
                processDiscardAndDetermineRemoval(
                    e, discardType, topLevelPackages, topLevelTargets, topLevelAspects);
            if (removeNode) {
              graph.remove(e.getKey());
            }
          });
    }
  }

  /** Signals whether nodes (or some internal node data) can be removed from the analysis cache. */
  private static boolean processDiscardAndDetermineRemoval(
      InMemoryNodeEntry entry,
      DiscardType discardType,
      ImmutableSet<PackageIdentifier> topLevelPackages,
      Collection<ConfiguredTarget> topLevelTargets,
      ImmutableSet<AspectKey> topLevelAspects) {
    SkyKey key = entry.getKey();
    SkyFunctionName functionName = key.functionName();
    if (discardType.discardsLoading()) {
      // Keep packages for top-level targets and aspects in memory to get the target from later.
      if (functionName.equals(SkyFunctions.PACKAGE) && topLevelPackages.contains(key.argument())) {
        return false;
      }
      if (LOADING_TYPES.contains(functionName)) {
        return true;
      }
    }
    if (discardType.discardsAnalysis()) {
      if (functionName.equals(SkyFunctions.CONFIGURED_TARGET)) {
        ConfiguredTargetValue ctValue = (ConfiguredTargetValue) entry.getValue();
        // ctValue may be null if target was not successfully analyzed.
        if (ctValue != null) {
          ConfiguredTarget configuredTarget = ctValue.getConfiguredTarget();
          if (configuredTarget == null) {
            return false; // It was already cleared.
          }
          if (!(ctValue instanceof ActionLookupValue)
              && discardType.discardsLoading()
              && !topLevelTargets.contains(configuredTarget)) {
            // If loading nodes are already being removed, removing these nodes doesn't hurt.
            // Morally we should always be able to remove these, since they're not used for
            // execution, but it leaves the graph inconsistent, and the --discard_analysis_cache
            // with --track_incremental_state case isn't worth optimizing for.
            return true;
          } else {
            ctValue.clear(!topLevelTargets.contains(configuredTarget));
          }
        }
      } else if (functionName.equals(SkyFunctions.ASPECT)) {
        AspectValue aspectValue = (AspectValue) entry.getValue();
        // aspectValue may be null if target was not successfully analyzed.
        if (aspectValue != null) {
          aspectValue.clear(!topLevelAspects.contains(key));
        }
      }
    }
    return false;
  }

  /** Tracks how long it takes to clear the analysis cache. */
  private SilentCloseable trackDiscardAnalysisCache(DiscardType discardType) {
    AutoProfiler profiler =
        GoogleAutoProfilerUtils.logged("discarding analysis cache " + discardType);
    return () -> {
      Duration d = Duration.ofNanos(profiler.completeAndGetElapsedTimeNanos());
      getEventBus().post(new AnalysisCacheClearEvent(d));
    };
  }

  /**
   * Saves memory by clearing analysis objects from Skyframe. Clears their data without deleting
   * them (they will be deleted on the next build). May also delete loading-phase objects from the
   * graph.
   */
  // VisibleForTesting but open-source annotation doesn't have productionVisibility option.
  public final void clearAnalysisCache(
      ImmutableSet<ConfiguredTarget> topLevelTargets, ImmutableSet<AspectKey> topLevelAspects) {
    this.analysisCacheCleared = true;
    clearAnalysisCacheImpl(topLevelTargets, topLevelAspects);
  }

  protected abstract void clearAnalysisCacheImpl(
      ImmutableSet<ConfiguredTarget> topLevelTargets, ImmutableSet<AspectKey> topLevelAspects);

  protected abstract void dropConfiguredTargetsNow(final ExtendedEventHandler eventHandler);

  protected final void deleteAnalysisNodes() {
    memoizingEvaluator.delete(
        shouldDeleteActionNodesWhenDroppingAnalysis()
            ? SkyframeExecutor::analysisInvalidatingPredicateWithActions
            : SkyframeExecutor::analysisInvalidatingPredicate);
  }

  // We delete any value that can hold an action -- all subclasses of ActionLookupKey.
  // Also remove ArtifactNestedSetValues to prevent memory leak (b/143940221).
  // Also BuildConfigurationKey to prevent memory leak (b/191875929).
  private static boolean analysisInvalidatingPredicate(SkyKey key) {
    return key instanceof ArtifactNestedSetKey
        || key instanceof ActionLookupKey
        || key instanceof BuildConfigurationKey;
  }

  // Also remove ActionLookupData since all such nodes depend on ActionLookupKey nodes and deleting
  // en masse is cheaper than deleting via graph traversal (b/192863968).
  private static boolean analysisInvalidatingPredicateWithActions(SkyKey key) {
    return analysisInvalidatingPredicate(key) || key instanceof ActionLookupData;
  }

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

  public void injectCoverageReportData(ImmutableList<ActionAnalysisMetadata> actions) {
    CoverageReportFunction.COVERAGE_REPORT_KEY.set(injectable(), actions);
  }

  private void setDefaultVisibility(RuleVisibility defaultVisibility) {
    PrecomputedValue.DEFAULT_VISIBILITY.set(injectable(), defaultVisibility);
  }

  private void setConfigSettingVisibilityPolicty(ConfigSettingVisibilityPolicy policy) {
    PrecomputedValue.CONFIG_SETTING_VISIBILITY_POLICY.set(injectable(), policy);
  }

  private void setStarlarkSemantics(StarlarkSemantics starlarkSemantics) {
    PrecomputedValue.STARLARK_SEMANTICS.set(injectable(), starlarkSemantics);
  }

  public void setBaselineConfiguration(BuildOptions buildOptions) {
    PrecomputedValue.BASELINE_CONFIGURATION.set(injectable(), buildOptions);
  }

  public void injectExtraPrecomputedValues(List<PrecomputedValue.Injected> extraPrecomputedValues) {
    for (PrecomputedValue.Injected injected : extraPrecomputedValues) {
      injected.inject(injectable());
    }
  }

  private void setShowLoadingProgress(boolean showLoadingProgressValue) {
    showLoadingProgress.set(showLoadingProgressValue);
  }

  protected void setCommandId(UUID commandId) {
    PrecomputedValue.BUILD_ID.set(injectable(), commandId);
  }

  /** Returns the build-info.txt and build-changelist.txt artifacts. */
  public ImmutableList<Artifact> getWorkspaceStatusArtifacts(ExtendedEventHandler eventHandler)
      throws InterruptedException {
    try (SilentCloseable c =
        Profiler.instance().profile("SkyframeExecutor.getWorkspaceStatusArtifact")) {
      // Should already be present, unless the user didn't request any targets for analysis.
      EvaluationResult<WorkspaceStatusValue> result =
          evaluate(
              ImmutableList.of(WorkspaceStatusValue.BUILD_INFO_KEY),
              /*keepGoing=*/ true,
              /*numThreads=*/ 1,
              eventHandler);
      WorkspaceStatusValue value = checkNotNull(result.get(WorkspaceStatusValue.BUILD_INFO_KEY));
      return ImmutableList.of(value.getStableArtifact(), value.getVolatileArtifact());
    }
  }

  public EventBus getEventBus() {
    return eventBus.get();
  }

  @VisibleForTesting
  ImmutableList<Root> getPathEntries() {
    return pkgLocator.get().getPathEntries();
  }

  public AtomicReference<PathPackageLocator> getPackageLocator() {
    return pkgLocator;
  }

  protected Differencer.Diff getDiff(
      TimestampGranularityMonitor tsgm,
      ModifiedFileSet modifiedFileSet,
      final Root pathEntry,
      int fsvcThreads)
      throws InterruptedException, AbruptExitException {
    if (modifiedFileSet.modifiedSourceFiles().isEmpty()) {
      return new ImmutableDiff(ImmutableList.of(), ImmutableMap.of());
    }

    // TODO(bazel-team): change ModifiedFileSet to work with RootedPaths instead of PathFragments.
    Collection<FileStateKey> dirtyFileStateSkyKeys =
        Collections2.transform(
            modifiedFileSet.modifiedSourceFiles(),
            pathFragment -> {
              checkState(
                  !pathFragment.isAbsolute(), "found absolute PathFragment: %s", pathFragment);
              return FileStateValue.key(RootedPath.toRootedPath(pathEntry, pathFragment));
            });

    return FileSystemValueCheckerInferringAncestors.getDiffWithInferredAncestors(
        tsgm,
        memoizingEvaluator.getInMemoryGraph(),
        dirtyFileStateSkyKeys,
        fsvcThreads,
        syscallCache,
        getSkyValueDirtinessCheckerForFiles());
  }

  /** Returns the {@link SkyValueDirtinessChecker} relevant for files. */
  @ForOverride
  protected SkyValueDirtinessChecker getSkyValueDirtinessCheckerForFiles() {
    return new FileDirtinessChecker();
  }

  /**
   * Deletes all loaded packages and their upwards transitive closure, forcing reevaluation of all
   * affected nodes.
   */
  public void clearLoadedPackages() {
    memoizingEvaluator.delete(k -> SkyFunctions.PACKAGE.equals(k.functionName()));
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
      PackageOptions packageOptions,
      BuildLanguageOptions buildLanguageOptions,
      UUID commandId,
      Map<String, String> clientEnv,
      QuiescingExecutors executors,
      TimestampGranularityMonitor tsgm) {
    checkNotNull(pkgLocator);
    checkNotNull(tsgm);
    setActive(true);

    this.tsgm.set(tsgm);
    setCommandId(commandId);
    this.clientEnv.set(clientEnv);

    setShowLoadingProgress(packageOptions.showLoadingProgress);
    setDefaultVisibility(packageOptions.defaultVisibility);
    if (!packageOptions.enforceConfigSettingVisibility) {
      setConfigSettingVisibilityPolicty(ConfigSettingVisibilityPolicy.LEGACY_OFF);
    } else {
      setConfigSettingVisibilityPolicty(
          packageOptions.configSettingPrivateDefaultVisibility
              ? ConfigSettingVisibilityPolicy.DEFAULT_STANDARD
              : ConfigSettingVisibilityPolicy.DEFAULT_PUBLIC);
    }

    StarlarkSemantics starlarkSemantics = getEffectiveStarlarkSemantics(buildLanguageOptions);
    setStarlarkSemantics(starlarkSemantics);
    setSiblingDirectoryLayout(
        starlarkSemantics.getBool(BuildLanguageOptions.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT));
    setPackageLocator(pkgLocator);

    this.pkgFactory.setGlobbingThreads(executors.globbingParallelism());
    this.pkgFactory.setMaxDirectoriesToEagerlyVisitInGlobbing(
        packageOptions.maxDirectoriesToEagerlyVisitInGlobbing);
    emittedEventState.clear();

    // Clear internal caches used by SkyFunctions used for package loading. If the SkyFunctions
    // never had a chance to restart (e.g. due to user interrupt, or an error in a --nokeep_going
    // build), these may have stale entries.
    bzlCompileCache.invalidateAll();

    numPackagesSuccessfullyLoaded.set(0);
    if (packageProgress != null) {
      packageProgress.reset();
    }

    // Reset the stateful SkyframeCycleReporter, which contains cycles from last run.
    cyclesReporter = createCyclesReporter();
    analysisCacheCleared = false;
  }

  private void setSiblingDirectoryLayout(boolean experimentalSiblingRepositoryLayout) {
    this.artifactFactory.setSiblingRepositoryLayout(experimentalSiblingRepositoryLayout);
  }

  public StarlarkSemantics getEffectiveStarlarkSemantics(
      BuildLanguageOptions buildLanguageOptions) {
    return buildLanguageOptions.toStarlarkSemantics();
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
      onPkgLocatorChange(oldLocator, pkgLocator);
    }
  }

  protected abstract void onPkgLocatorChange(
      PathPackageLocator oldLocator, PathPackageLocator pkgLocator);

  public SkyframeBuildView getSkyframeBuildView() {
    return skyframeBuildView;
  }

  /** Sets whether this build is done with --experimental_merged_skyframe_analysis_execution. */
  public void setMergedSkyframeAnalysisExecutionSupplier(
      Supplier<Boolean> mergedSkyframeAnalysisExecutionSupplier) {
    this.mergedSkyframeAnalysisExecutionSupplier = mergedSkyframeAnalysisExecutionSupplier;
  }

  boolean isMergedSkyframeAnalysisExecution() {
    return mergedSkyframeAnalysisExecutionSupplier != null
        && mergedSkyframeAnalysisExecutionSupplier.get();
  }

  /** Sets the eventBus to use for posting events. */
  public void setEventBus(@Nullable EventBus eventBus) {
    this.eventBus.set(eventBus);
  }

  public void setClientEnv(Map<String, String> clientEnv) {
    this.skyframeActionExecutor.setClientEnv(clientEnv);
  }

  /** Sets the path for action log buffers. */
  public void setActionOutputRoot(Path actionOutputRoot) {
    checkNotNull(actionOutputRoot);
    this.actionLogBufferPathGenerator = new ActionLogBufferPathGenerator(actionOutputRoot);
    this.skyframeActionExecutor.setActionLogBufferPathGenerator(actionLogBufferPathGenerator);
  }

  private void setRemoteExecutionEnabled(boolean enabled) {
    PrecomputedValue.REMOTE_EXECUTION_ENABLED.set(injectable(), enabled);
  }

  /** Called when a top-level configuration is determined. */
  protected void setTopLevelConfiguration(BuildConfigurationValue topLevelConfiguration) {}

  /** Asks the Skyframe evaluator to build a {@link BuildConfigurationValue}. */
  public BuildConfigurationValue createConfiguration(
      ExtendedEventHandler eventHandler, BuildOptions buildOptions, boolean keepGoing)
      throws InvalidConfigurationException {

    if (configuredTargetProgress != null) {
      configuredTargetProgress.reset();
    }

    BuildConfigurationValue topLevelTargetConfig =
        getConfiguration(eventHandler, buildOptions, keepGoing);

    // TODO(gregce): cache invalid option errors in BuildConfigurationFunction, then use a dedicated
    // accessor (i.e. not the event handler) to trigger the exception below.
    ErrorSensingEventHandler<Void> nosyEventHandler =
        ErrorSensingEventHandler.withoutPropertyValueTracking(eventHandler);
    topLevelTargetConfig.reportInvalidOptions(nosyEventHandler);
    if (nosyEventHandler.hasErrors()) {
      throw new InvalidConfigurationException(
          "Build options are invalid", Code.INVALID_BUILD_OPTIONS);
    }
    return topLevelTargetConfig;
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
      ImmutableSet<AspectKey> aspects,
      Set<ConfiguredTarget> parallelTests,
      Set<ConfiguredTarget> exclusiveTests,
      OptionsProvider options,
      ActionCacheChecker actionCacheChecker,
      @Nullable EvaluationProgressReceiver executionProgressReceiver,
      TopLevelArtifactContext topLevelArtifactContext)
      throws InterruptedException, AbruptExitException {
    checkActive();
    checkState(actionLogBufferPathGenerator != null);

    deleteActionsIfRemoteOptionsChanged(options);
    try (SilentCloseable c =
        Profiler.instance().profile("skyframeActionExecutor.prepareForExecution")) {
      prepareSkyframeActionExecutorForExecution(reporter, executor, options, actionCacheChecker);
    }

    resourceManager.resetResourceUsage();
    try {
      setExecutionProgressReceiver(executionProgressReceiver);
      Iterable<TargetCompletionValue.TargetCompletionKey> targetKeys =
          TargetCompletionValue.keys(
              targetsToBuild, topLevelArtifactContext, Sets.union(parallelTests, exclusiveTests));
      Iterable<SkyKey> aspectKeys = AspectCompletionValue.keys(aspects, topLevelArtifactContext);
      Iterable<SkyKey> testKeys =
          TestCompletionValue.keys(
              parallelTests, topLevelArtifactContext, /*exclusiveTesting=*/ false);
      EvaluationContext evaluationContext =
          newEvaluationContextBuilder()
              .setKeepGoing(options.getOptions(KeepGoingOption.class).keepGoing)
              .setParallelism(options.getOptions(BuildRequestOptions.class).jobs)
              .setEventHandler(reporter)
              .setExecutionPhase()
              .build();
      return memoizingEvaluator.evaluate(
          Iterables.concat(Artifact.keys(artifactsToBuild), targetKeys, aspectKeys, testKeys),
          evaluationContext);
    } finally {
      // Also releases thread locks.
      resourceManager.resetResourceUsage();
      cleanUpAfterSingleEvaluationWithActionExecution(reporter);
    }
  }

  public void setExecutionProgressReceiver(
      @Nullable EvaluationProgressReceiver executionProgressReceiver) {
    progressReceiver.executionProgressReceiver = executionProgressReceiver;
  }

  public void prepareSkyframeActionExecutorForExecution(
      Reporter reporter,
      Executor executor,
      OptionsProvider options,
      ActionCacheChecker actionCacheChecker) {
    skyframeActionExecutor.prepareForExecution(
        reporter,
        executor,
        options,
        actionCacheChecker,
        outputService,
        tracksStateForIncrementality());
  }

  /** Asks the Skyframe evaluator to run a single exclusive test. */
  public EvaluationResult<?> runExclusiveTest(
      Reporter reporter,
      ResourceManager resourceManager,
      Executor executor,
      ConfiguredTarget exclusiveTest,
      OptionsProvider options,
      ActionCacheChecker actionCacheChecker,
      TopLevelArtifactContext topLevelArtifactContext)
      throws InterruptedException {
    checkActive();
    checkState(actionLogBufferPathGenerator != null);

    try (SilentCloseable c =
        Profiler.instance().profile("skyframeActionExecutor.prepareForExecution")) {
      prepareSkyframeActionExecutorForExecution(reporter, executor, options, actionCacheChecker);
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
      // Also releases thread locks.
      resourceManager.resetResourceUsage();
      cleanUpAfterSingleEvaluationWithActionExecution(reporter);
    }
  }

  public EvaluationResult<SkyValue> runExclusiveTestSkymeld(
      ExtendedEventHandler eventHandler,
      ResourceManager resourceManager,
      SkyKey testCompletionKey,
      boolean keepGoing,
      int numThreads)
      throws InterruptedException {
    checkActive();
    checkState(actionLogBufferPathGenerator != null);

    resourceManager.resetResourceUsage();
    try {
      return evaluate(ImmutableSet.of(testCompletionKey), keepGoing, numThreads, eventHandler);
    } finally {
      resourceManager.resetResourceUsage();
    }
  }

  @VisibleForTesting
  public void prepareBuildingForTestingOnly(
      Reporter reporter, Executor executor, OptionsProvider options, ActionCacheChecker checker) {
    prepareSkyframeActionExecutorForExecution(reporter, executor, options, checker);
  }

  public void deleteActionsIfRemoteOptionsChanged(OptionsProvider options)
      throws AbruptExitException {
    RemoteOptions remoteOptions = options.getOptions(RemoteOptions.class);
    Map<String, String> remoteDefaultExecProperties;
    try {
      remoteDefaultExecProperties =
          remoteOptions != null
              ? remoteOptions.getRemoteDefaultExecProperties()
              : ImmutableMap.of();
    } catch (UserExecException e) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(e.getMessage())
                  .setRemoteOptions(
                      FailureDetails.RemoteOptions.newBuilder()
                          .setCode(
                              FailureDetails.RemoteOptions.Code
                                  .REMOTE_DEFAULT_EXEC_PROPERTIES_LOGIC_ERROR)
                          .build())
                  .build()),
          e);
    }
    boolean needsDeletion =
        lastRemoteDefaultExecProperties != null
            && !remoteDefaultExecProperties.equals(lastRemoteDefaultExecProperties);
    lastRemoteDefaultExecProperties = remoteDefaultExecProperties;

    boolean remoteCacheEnabled = remoteOptions != null && remoteOptions.isRemoteCacheEnabled();
    // If we have remote metadata from last build, and the remote cache is not
    // enabled in this build, invalidate actions since they can't download those
    // remote files.
    //
    // TODO(chiwang): Re-evaluate this after action rewinding is implemented in
    //  Bazel since we can treat that case as lost inputs.
    if (lastRemoteOutputsMode != RemoteOutputsMode.ALL) {
      needsDeletion |=
          lastRemoteCacheEnabled != null && lastRemoteCacheEnabled && !remoteCacheEnabled;
    }
    lastRemoteCacheEnabled = remoteCacheEnabled;
    lastRemoteOutputsMode =
        remoteOptions != null ? remoteOptions.remoteOutputsMode : RemoteOutputsMode.ALL;

    if (needsDeletion) {
      memoizingEvaluator.delete(k -> SkyFunctions.ACTION_EXECUTION.equals(k.functionName()));
    }
  }

  EvaluationResult<SkyValue> targetPatterns(
      Iterable<? extends SkyKey> patternSkyKeys,
      int numThreads,
      boolean keepGoing,
      ExtendedEventHandler eventHandler)
      throws InterruptedException {
    checkActive();
    EvaluationContext evaluationContext =
        newEvaluationContextBuilder()
            .setKeepGoing(keepGoing)
            .setParallelism(numThreads)
            .setEventHandler(eventHandler)
            .build();
    return memoizingEvaluator.evaluate(patternSkyKeys, evaluationContext);
  }

  @Nullable
  public BuildConfigurationValue getConfiguration(
      ExtendedEventHandler eventHandler, @Nullable BuildConfigurationKey configurationKey) {
    if (configurationKey == null) {
      return null;
    }
    return (BuildConfigurationValue)
        evaluateSkyKeys(eventHandler, ImmutableList.of(configurationKey)).get(configurationKey);
  }
  /**
   * Returns the configurations corresponding to the given sets of build options. Output order is
   * the same as input order.
   *
   * @throws InvalidConfigurationException if any build options produces an invalid configuration
   */
  // TODO(ulfjack): Remove this legacy method after switching to the Skyframe-based implementation.
  public BuildConfigurationValue getConfiguration(
      ExtendedEventHandler eventHandler, BuildOptions buildOptions, boolean keepGoing)
      throws InvalidConfigurationException {
    // Prepare the Skyframe inputs.

    PlatformMappingValue platformMappingValue = getPlatformMappingValue(eventHandler, buildOptions);

    SkyKey configSkyKey = toConfigurationKey(platformMappingValue, buildOptions);

    // Skyframe-evaluate the configurations and throw errors if any.
    EvaluationResult<SkyValue> evalResult =
        evaluateSkyKeys(eventHandler, ImmutableList.of(configSkyKey), keepGoing);
    if (evalResult.hasError()) {
      Map.Entry<SkyKey, ErrorInfo> firstError = Iterables.get(evalResult.errorMap().entrySet(), 0);
      ErrorInfo error = firstError.getValue();
      Throwable e = error.getException();
      // Wrap loading failed exceptions
      if (e instanceof NoSuchThingException) {
        e = new InvalidConfigurationException(((NoSuchThingException) e).getDetailedExitCode(), e);
      } else if (e == null && !error.getCycleInfo().isEmpty()) {
        cyclesReporter.reportCycles(error.getCycleInfo(), firstError.getKey(), eventHandler);
        e =
            new InvalidConfigurationException(
                "cannot load build configuration because of this cycle", Code.CYCLE);
      }
      if (e != null) {
        Throwables.throwIfInstanceOf(e, InvalidConfigurationException.class);
      }
      throw new IllegalStateException("Unknown error during configuration creation evaluation", e);
    }

    // Prepare and return the results.
    return (BuildConfigurationValue) evalResult.get(configSkyKey);
  }

  public Map<BuildConfigurationKey, BuildConfigurationValue> getConfigurations(
      ExtendedEventHandler eventHandler, Collection<BuildConfigurationKey> keys) {
    EvaluationResult<SkyValue> evaluationResult = evaluateSkyKeys(eventHandler, keys);
    return keys.stream()
        .collect(
            toImmutableMap(
                Functions.identity(), key -> (BuildConfigurationValue) evaluationResult.get(key)));
  }

  /** Returns every {@link BuildConfigurationKey} in the graph. */
  public Collection<SkyKey> getTransitiveConfigurationKeys() {
    return memoizingEvaluator.getDoneValues().keySet().stream()
        .filter(key -> SkyFunctions.BUILD_CONFIGURATION.equals(key.functionName()))
        .collect(toImmutableList());
  }

  private PlatformMappingValue getPlatformMappingValue(
      ExtendedEventHandler eventHandler, BuildOptions referenceBuildOptions)
      throws InvalidConfigurationException {
    PathFragment platformMappingPath =
        referenceBuildOptions.hasNoConfig()
            ? null
            : referenceBuildOptions.get(PlatformOptions.class).platformMappings;

    PlatformMappingValue.Key platformMappingKey =
        PlatformMappingValue.Key.create(platformMappingPath);
    EvaluationResult<SkyValue> evaluationResult =
        evaluateSkyKeys(eventHandler, ImmutableSet.of(platformMappingKey));
    if (evaluationResult.hasError()) {
      throw new InvalidConfigurationException(
          Code.PLATFORM_MAPPING_EVALUATION_FAILURE, evaluationResult.getError().getException());
    }
    return (PlatformMappingValue) evaluationResult.get(platformMappingKey);
  }

  private static BuildConfigurationKey toConfigurationKey(
      PlatformMappingValue platformMappingValue, BuildOptions toOption)
      throws InvalidConfigurationException {
    try {
      return BuildConfigurationKey.withPlatformMapping(platformMappingValue, toOption);
    } catch (OptionsParsingException e) {
      throw new InvalidConfigurationException(Code.INVALID_BUILD_OPTIONS, e);
    }
  }

  /**
   * Evaluates the given sky keys, blocks, and returns their evaluation results. Fails fast on the
   * first evaluation error.
   */
  private EvaluationResult<SkyValue> evaluateSkyKeys(
      final ExtendedEventHandler eventHandler, final Iterable<? extends SkyKey> skyKeys) {
    return evaluateSkyKeys(eventHandler, skyKeys, false);
  }

  /** Evaluates sky keys that require action execution and returns their evaluation results. */
  public EvaluationResult<SkyValue> evaluateSkyKeysWithExecution(
      final Reporter reporter,
      final Executor executor,
      final Iterable<? extends SkyKey> skyKeys,
      final OptionsProvider options,
      final ActionCacheChecker actionCacheChecker) {

    prepareSkyframeActionExecutorForExecution(reporter, executor, options, actionCacheChecker);
    try {
      return evaluateSkyKeys(
          reporter, skyKeys, options.getOptions(KeepGoingOption.class).keepGoing);
    } finally {
      cleanUpAfterSingleEvaluationWithActionExecution(reporter);
    }
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
              () -> {
                try (var closer = new EnableAnalysisScope()) {
                  synchronized (valueLookupLock) {
                    return evaluate(
                        skyKeys, keepGoing, /* numThreads= */ DEFAULT_THREAD_COUNT, eventHandler);
                  }
                }
              });
    } catch (Exception e) {
      throw new IllegalStateException(e); // Should never happen.
    }
    return result;
  }

  private class EnableAnalysisScope implements AutoCloseable {
    private EnableAnalysisScope() {
      skyframeBuildView.enableAnalysis(true);
    }

    @Override
    public void close() {
      skyframeBuildView.enableAnalysis(false);
    }
  }

  /** Invalidates SkyFrame values that may have failed for transient reasons. */
  public abstract void invalidateTransientErrors();

  /** Configures a given set of configured targets. */
  @CanIgnoreReturnValue
  protected ConfigureTargetsResult configureTargets(
      ExtendedEventHandler eventHandler,
      ImmutableMap<Label, Target> labelToTargetMap,
      ImmutableList<ConfiguredTargetKey> configuredTargetKeys,
      ImmutableList<TopLevelAspectsKey> topLevelAspectKeys,
      boolean keepGoing,
      QuiescingExecutors executors)
      throws InterruptedException {
    checkActive();

    eventHandler.post(new ConfigurationPhaseStartedEvent(configuredTargetProgress));
    EvaluationContext evaluationContext =
        newEvaluationContextBuilder()
            .setParallelism(executors.analysisParallelism())
            .setKeepGoing(keepGoing)
            .setExecutor(executors.getAnalysisExecutor())
            .setEventHandler(eventHandler)
            .build();
    EvaluationResult<ActionLookupValue> result =
        memoizingEvaluator.evaluate(
            Iterables.concat(configuredTargetKeys, topLevelAspectKeys), evaluationContext);
    syscallCache.noteAnalysisPhaseEnded();

    var targetsWithConfiguration =
        ImmutableList.<TargetAndConfiguration>builderWithExpectedSize(configuredTargetKeys.size());
    ImmutableSet.Builder<ConfiguredTarget> configuredTargets = ImmutableSet.builder();
    ImmutableMap.Builder<AspectKey, ConfiguredAspect> aspects = ImmutableMap.builder();

    WalkableGraph graph = result.getWalkableGraph();
    for (ConfiguredTargetKey key : configuredTargetKeys) {
      var value = (ConfiguredTargetValue) result.get(key);
      if (value == null) {
        continue;
      }
      ConfiguredTarget configuredTarget = value.getConfiguredTarget();
      configuredTargets.add(configuredTarget);

      Target target = labelToTargetMap.get(key.getLabel());
      BuildConfigurationValue configuration =
          getConfigurationFromGraph(graph, configuredTarget.getConfigurationKey());
      targetsWithConfiguration.add(new TargetAndConfiguration(target, configuration));
      eventHandler.post(new TargetConfiguredEvent(target, configuration));
    }

    for (TopLevelAspectsKey key : topLevelAspectKeys) {
      TopLevelAspectsValue value = (TopLevelAspectsValue) result.get(key);
      if (value == null) {
        continue; // Skip aspects that couldn't be applied to targets.
      }
      for (AspectValue aspectValue : value.getTopLevelAspectsValues()) {
        AspectKey aspectKey = aspectValue.getKey();
        aspects.put(aspectKey, aspectValue);
        BuildConfigurationValue configuration =
            getConfigurationFromGraph(graph, aspectKey.getConfigurationKey());
        eventHandler.post(
            new AspectConfiguredEvent(
                aspectKey.getLabel(),
                /* aspectClassName= */ aspectKey.getAspectClass().getName(),
                /* aspectDescription= */ aspectKey.getAspectDescriptor().getDescription(),
                configuration));
      }
    }

    return new AutoValue_SkyframeExecutor_ConfigureTargetsResult(
        result,
        configuredTargets.build(),
        aspects.buildOrThrow(),
        targetsWithConfiguration.build(),
        getPackageRoots());
  }

  @ForOverride
  protected PackageRoots getPackageRoots() {
    return new MapAsPackageRoots(collectPackageRoots());
  }

  @Nullable
  private static BuildConfigurationValue getConfigurationFromGraph(
      WalkableGraph graph, @Nullable BuildConfigurationKey key) throws InterruptedException {
    return key == null ? null : (BuildConfigurationValue) graph.getValue(key);
  }

  /** Result of a call to {@link #configureTargets}. */
  @AutoValue
  protected abstract static class ConfigureTargetsResult {
    public abstract EvaluationResult<ActionLookupValue> evaluationResult();

    public abstract ImmutableSet<ConfiguredTarget> configuredTargets();

    public abstract ImmutableMap<AspectKey, ConfiguredAspect> aspects();

    public abstract ImmutableList<TargetAndConfiguration> targetsWithConfiguration();

    public abstract PackageRoots packageRoots();
  }

  /** Returns a map of collected package names to root paths. */
  private ImmutableMap<PackageIdentifier, Root> collectPackageRoots() {
    Map<PackageIdentifier, Root> roots = new ConcurrentHashMap<>();
    memoizingEvaluator
        .getInMemoryGraph()
        .parallelForEach(
            nodeEntry -> {
              SkyKey key = nodeEntry.getKey();
              if (key instanceof PackageIdentifier && nodeEntry.isDone()) {
                PackageValue packageValue = (PackageValue) nodeEntry.getValue();
                if (packageValue != null) { // Null for errors e.g. "no such package"
                  Optional<Root> sourceRoot = packageValue.getPackage().getSourceRoot();
                  if (sourceRoot.isPresent()) {
                    roots.put(
                        (PackageIdentifier) key,
                        maybeTransformSourceRootForExecrootSymlinkCreation(sourceRoot.get()));
                  }
                }
              }
            });
    return ImmutableMap.copyOf(roots);
  }

  /** Returns a possibly transformed source root of a package for execroot symlink creation. */
  @ForOverride
  protected Root maybeTransformSourceRootForExecrootSymlinkCreation(Root sourceRoot) {
    return sourceRoot;
  }

  void clearSyscallCache() {
    syscallCache.clear();
  }

  /**
   * Evaluates the given collections of CT/Aspect BuildDriverKeys. This is part of
   * https://github.com/bazelbuild/bazel/issues/14057, internal: b/147350683.
   */
  EvaluationResult<SkyValue> evaluateBuildDriverKeys(
      ExtendedEventHandler eventHandler,
      Set<BuildDriverKey> buildDriverCTKeys,
      Set<BuildDriverKey> buildDriverAspectKeys,
      ImmutableList<Artifact> workspaceStatusArtifacts,
      boolean keepGoing,
      int executionParallelism,
      QuiescingExecutor executor,
      Supplier<Boolean> shouldCheckForConflicts)
      throws InterruptedException {
    checkActive();
    try {
      buildDriverFunction.setShouldCheckForConflict(shouldCheckForConflicts);
      if (shouldCheckForConflicts.get()) {
        initializeSkymeldConflictFindingStates();
      }
      eventHandler.post(new ConfigurationPhaseStartedEvent(configuredTargetProgress));
      // For the workspace status actions.
      eventHandler.post(SomeExecutionStartedEvent.create());
      EvaluationContext evaluationContext =
          newEvaluationContextBuilder()
              .setKeepGoing(keepGoing)
              .setParallelism(executionParallelism)
              .setExecutor(executor)
              .setEventHandler(eventHandler)
              .setMergingSkyframeAnalysisExecutionPhases(true)
              .build();
      return memoizingEvaluator.evaluate(
          Iterables.concat(
              buildDriverCTKeys, buildDriverAspectKeys, Artifact.keys(workspaceStatusArtifacts)),
          evaluationContext);
    } finally {
      clearSyscallCache();
    }
  }

  /** Called after a single Skyframe evaluation that involves action execution. */
  private void cleanUpAfterSingleEvaluationWithActionExecution(ExtendedEventHandler eventHandler) {
    setExecutionProgressReceiver(null);

    skyframeActionExecutor.executionOver();
    actionExecutionFunction.complete(eventHandler);
  }

  /**
   * Clears the various states required for execution after ALL action execution in the build is
   * done.
   */
  public void clearExecutionStatesSkymeld(ExtendedEventHandler eventHandler) {
    Preconditions.checkNotNull(watchdog).stop();
    watchdog = null;
    cleanUpAfterSingleEvaluationWithActionExecution(eventHandler);
    statusReporterRef.get().unregisterFromEventBus();
    setActionExecutionProgressReportingObjects(null, null, null);
  }

  /**
   * Checks the given action lookup values for action conflicts. Values satisfying the returned
   * predicate are known to be transitively error-free from action conflicts or other analysis
   * failures. {@link #resetActionConflictsStoredInSkyframe} must be called after this to free
   * memory coming from this call.
   */
  TopLevelActionConflictReport filterActionConflictsForConfiguredTargetsAndAspects(
      ExtendedEventHandler eventHandler,
      Iterable<ActionLookupKey> keys,
      ImmutableMap<ActionAnalysisMetadata, ArtifactConflictFinder.ConflictException>
          actionConflicts,
      TopLevelArtifactContext topLevelArtifactContext)
      throws InterruptedException {
    checkActive();
    ACTION_CONFLICTS.set(injectable(), actionConflicts);
    // This work is CPU-bound, so use the number of available processors.
    EvaluationResult<ActionLookupConflictFindingValue> result =
        evaluate(
            TopLevelActionLookupConflictFindingFunction.keys(keys, topLevelArtifactContext),
            /*keepGoing=*/ true,
            /*numThreads=*/ ResourceUsage.getAvailableProcessors(),
            eventHandler);

    // Remove top-level action-conflict detection values for memory efficiency. Non-top-level ones
    // are removed below. We are OK with this mini-phase being non-incremental as the failure mode
    // of action conflict is rare.
    memoizingEvaluator.delete(
        SkyFunctionName.functionIs(SkyFunctions.TOP_LEVEL_ACTION_LOOKUP_CONFLICT_FINDING));
    return new TopLevelActionConflictReport(result, topLevelArtifactContext);
  }

  /**
   * Encapsulation of the result of #filterActionConflictsForConfiguredTargetsAndAspects() allowing
   * callers to determine which top-level keys did not have analysis errors and retrieve the
   * ConflictException for those that keys that specifically have conflicts.
   */
  static final class TopLevelActionConflictReport {

    public final EvaluationResult<ActionLookupConflictFindingValue> result;
    private final TopLevelArtifactContext topLevelArtifactContext;

    TopLevelActionConflictReport(
        EvaluationResult<ActionLookupConflictFindingValue> result,
        TopLevelArtifactContext topLevelArtifactContext) {
      this.result = result;
      this.topLevelArtifactContext = topLevelArtifactContext;
    }

    boolean isErrorFree(ActionLookupKey k) {
      return result.get(
              TopLevelActionLookupConflictFindingFunction.Key.create(k, topLevelArtifactContext))
          != null;
    }

    /**
     * Get the ConflictException produced for the given ActionLookupKey. Will throw if the given key
     * {@link #isErrorFree is error-free}.
     */
    Optional<ConflictException> getConflictException(ActionLookupKey k) {
      ErrorInfo errorInfo =
          result.getError(
              TopLevelActionLookupConflictFindingFunction.Key.create(k, topLevelArtifactContext));
      Exception e = errorInfo.getException();
      return Optional.ofNullable(e instanceof ConflictException ? (ConflictException) e : null);
    }
  }

  /**
   * Clears all action conflicts stored in skyframe that were discovered by a call to {@link
   * #filterActionConflictsForConfiguredTargetsAndAspects}.
   *
   * <p>This function must be called after a call to {@link
   * #filterActionConflictsForConfiguredTargetsAndAspects}, either directly (in the case of
   * no-keep_going evaluations) or indirectly by {@link #filterActionConflictsForTopLevelArtifacts}
   * in keep_going evaluations.
   */
  void resetActionConflictsStoredInSkyframe() {
    memoizingEvaluator.delete(
        SkyFunctionName.functionIs(SkyFunctions.ACTION_LOOKUP_CONFLICT_FINDING));
  }

  public void resetBuildDriverFunction() {
    buildDriverFunction.resetStates();
  }

  // Initialize the various conflict-finding states. These are good for 1 invocation.
  private void initializeSkymeldConflictFindingStates() {
    incrementalArtifactConflictFinder =
        IncrementalArtifactConflictFinder.createWithActionGraph(
            new MapBasedActionGraph(actionKeyContext));
    conflictFreeActionLookupKeysGlobalSet = Sets.newConcurrentHashSet();
    incrementalTransitiveActionLookupKeysCollector =
        new TransitiveActionLookupKeysCollector(
            SkyframeExecutorWrappingWalkableGraph.of(this), conflictFreeActionLookupKeysGlobalSet);
  }

  /** Clear the incremental conflict finding states to save memory. */
  public void clearIncrementalArtifactConflictFindingStates() {
    // Create a local ref for shutting down, in case there's a race.
    IncrementalArtifactConflictFinder localRef = incrementalArtifactConflictFinder;
    if (localRef != null) {
      localRef.shutdown();
    }
    incrementalArtifactConflictFinder = null;
    conflictFreeActionLookupKeysGlobalSet = Sets.newConcurrentHashSet();
    incrementalTransitiveActionLookupKeysCollector = null;
  }

  public IncrementalArtifactConflictFinder getIncrementalArtifactConflictFinder() {
    return incrementalArtifactConflictFinder;
  }

  private Set<ActionLookupKey> getConflictFreeActionLookupKeysGlobalSet() {
    return conflictFreeActionLookupKeysGlobalSet;
  }

  /**
   * Checks the action lookup values owning the given artifacts for action conflicts. Artifacts
   * satisfying the returned predicate are known to be transitively free from action conflicts.
   * {@link #filterActionConflictsForConfiguredTargetsAndAspects} must be called before this is
   * called in order to populate the known action conflicts.
   *
   * <p>This method is only called in keep-going mode, since otherwise any known action conflicts
   * will immediately fail the build.
   */
  public Predicate<Artifact> filterActionConflictsForTopLevelArtifacts(
      ExtendedEventHandler eventHandler, Collection<Artifact> artifacts)
      throws InterruptedException {
    checkActive();
    // This work is CPU-bound, so use the number of available processors.
    EvaluationResult<ActionLookupConflictFindingValue> result =
        evaluate(
            Iterables.transform(artifacts, ActionLookupConflictFindingValue::key),
            /*keepGoing=*/ true,
            /*numThreads=*/ ResourceUsage.getAvailableProcessors(),
            eventHandler);

    // Remove remaining action-conflict detection values immediately for memory efficiency.
    resetActionConflictsStoredInSkyframe();

    return a -> result.get(ActionLookupConflictFindingValue.key(a)) != null;
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
  public final EvaluationResult<SkyValue> prepareAndGet(
      Set<SkyKey> roots, EvaluationContext evaluationContext) throws InterruptedException {
    EvaluationContext evaluationContextToUse =
        evaluationContext.builder().setKeepGoing(/*keepGoing=*/ true).build();
    return memoizingEvaluator.evaluate(roots, evaluationContextToUse);
  }

  public Optional<UniverseScope> maybeGetHardcodedUniverseScope() {
    return Optional.empty();
  }

  /** Returns the generating action of a given artifact ({@code null} if it's a source artifact). */
  @Nullable
  private ActionAnalysisMetadata getGeneratingAction(
      ExtendedEventHandler eventHandler, Artifact artifact) throws InterruptedException {
    if (artifact.isSourceArtifact()) {
      return null;
    }

    ActionLookupData generatingActionKey =
        ((Artifact.DerivedArtifact) artifact).getGeneratingActionKey();

    ActionLookupKey lookupKey = generatingActionKey.getActionLookupKey();

    synchronized (valueLookupLock) {
      // Note that this will crash (attempting to run a configured target value builder after
      // analysis) after a failed --nokeep_going analysis in which the configured target that
      // failed was a (transitive) dependency of the configured target that should generate
      // this action. We don't expect callers to query generating actions in such cases.
      EvaluationResult<ActionLookupValue> result =
          evaluate(
              ImmutableList.of(lookupKey),
              /* keepGoing= */ false,
              /* numThreads= */ ResourceUsage.getAvailableProcessors(),
              eventHandler);
      if (result.hasError()) {
        return null;
      }
      ActionLookupValue actionLookupValue = result.get(lookupKey);
      return actionLookupValue.getActions().get(generatingActionKey.getActionIndex());
    }
  }

  /**
   * Returns an action graph.
   *
   * <p>For legacy compatibility only.
   */
  public ActionGraph getActionGraph(final ExtendedEventHandler eventHandler) {
    return artifact -> {
      try {
        return callUninterruptibly(
            () -> SkyframeExecutor.this.getGeneratingAction(eventHandler, artifact));
      } catch (Exception e) {
        throw new IllegalStateException(
            "Error getting generating action: " + artifact.prettyPrint(), e);
      }
    };
  }

  public PackageManager getPackageManager() {
    return packageManager;
  }

  public QueryTransitivePackagePreloader getQueryTransitivePackagePreloader() {
    return queryTransitivePackagePreloader;
  }

  @VisibleForTesting
  public TargetPatternPreloader newTargetPatternPreloader() {
    return new SkyframeTargetPatternEvaluator(this);
  }

  public ActionKeyContext getActionKeyContext() {
    return actionKeyContext;
  }

  // TODO(janakr): Is there a better place for this?
  public final DigestHashFunction getDigestFunction() {
    return fileSystem.getDigestFunction();
  }

  /** Exception thrown when {@link #getDoneSkyValueForIntrospection} fails. */
  public static final class FailureToRetrieveIntrospectedValueException extends Exception {
    private FailureToRetrieveIntrospectedValueException(String message) {
      super(message);
    }

    private FailureToRetrieveIntrospectedValueException(
        String message, InterruptedException cause) {
      super(message, cause);
    }
  }

  /**
   * Returns the value of a node that the caller knows to be done. May be called intra-evaluation.
   * Null values and interrupts are unexpected, and will cause a {@link
   * FailureToRetrieveIntrospectedValueException}. Callers should handle gracefully, probably via
   * {@link BugReport}.
   */
  @ThreadSafety.ThreadSafe
  public SkyValue getDoneSkyValueForIntrospection(SkyKey key)
      throws FailureToRetrieveIntrospectedValueException {
    NodeEntry entry;
    try {
      entry = memoizingEvaluator.getExistingEntryAtCurrentlyEvaluatingVersion(key);
    } catch (InterruptedException e) {
      throw new FailureToRetrieveIntrospectedValueException(
          "Unexpected interrupt when fetching " + key, e);
    }
    if (entry == null || !entry.isDone()) {
      throw new FailureToRetrieveIntrospectedValueException(
          "Entry for " + key + " not found or null: " + entry);
    }
    SkyValue value;
    try {
      value = entry.getValue();
    } catch (InterruptedException e) {
      throw new FailureToRetrieveIntrospectedValueException(
          "Entry for " + key + " did not have locally present value: " + entry, e);
    }
    if (value == null) {
      throw new FailureToRetrieveIntrospectedValueException(
          "Entry for " + key + " had null value: " + entry);
    }
    return value;
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
      ImmutableList<SkyKey> keys = ImmutableList.of(pkgName);
      EvaluationResult<PackageValue> result;
      synchronized (valueLookupLock) {
        // Loading a single package shouldn't be too bad to do in keep_going mode even if the build
        // overall is in nokeep_going mode: the worst that happens is we parse some unnecessary
        // .bzl files.
        result =
            evaluate(keys, /*keepGoing=*/ true, /*numThreads=*/ DEFAULT_THREAD_COUNT, eventHandler);
      }
      ErrorInfo error = result.getError(pkgName);
      if (error != null) {
        if (!error.getCycleInfo().isEmpty()) {
          cyclesReporter.reportCycles(result.getError().getCycleInfo(), pkgName, eventHandler);
          // This can only happen if a package is freshly loaded outside of the target parsing or
          // loading phase
          throw new BuildFileContainsErrorsException(
              pkgName, "Cycle encountered while loading package " + pkgName);
        }
        Throwable e = checkNotNull(error.getException(), "%s %s", pkgName, error);
        // PackageFunction should be catching, swallowing, and rethrowing all transitive errors as
        // NoSuchPackageExceptions or constructing packages with errors, since we're in keep_going
        // mode.
        Throwables.throwIfInstanceOf(e, NoSuchPackageException.class);
        throw new IllegalStateException(
            "Unexpected Exception type from PackageValue for '"
                + pkgName
                + "'' with error: "
                + error,
            e);
      }
      return result.get(pkgName).getPackage();
    }

    /** Returns whether the given package should be consider deleted and thus should be ignored. */
    public boolean isPackageDeleted(PackageIdentifier packageName) {
      return deletedPackages.get().contains(packageName);
    }

    PackageLookupValue getPackageLookupValue(PackageIdentifier pkgName) {
      try {
        return (PackageLookupValue)
            memoizingEvaluator.getExistingValue(PackageLookupValue.key(pkgName));
      } catch (InterruptedException e) {
        throw new IllegalStateException(
            String.format(
                "Evaluator %s should not be interruptible (%s)", memoizingEvaluator, pkgName),
            e);
      }
    }

    void dumpPackages(PrintStream out) {
      SkyframeExecutor.this.dumpPackages(out);
    }
  }

  public MemoizingEvaluator getEvaluator() {
    return memoizingEvaluator;
  }

  /**
   * Initializes and syncs the graph with the given options, readying it for the next evaluation.
   *
   * <p>At a minimum, {@link PackageOptions} and {@link BuildLanguageOptions} are expected to be
   * present in the given {@link OptionsProvider}.
   *
   * <p>Returns precomputed information about the workspace if it is available at this stage. This
   * is an optimization allowing implementations which have such information to make it available
   * early in the build.
   */
  @Nullable
  @CanIgnoreReturnValue
  public WorkspaceInfoFromDiff sync(
      ExtendedEventHandler eventHandler,
      PathPackageLocator pathPackageLocator,
      UUID commandId,
      Map<String, String> clientEnv,
      Map<String, String> repoEnvOption,
      TimestampGranularityMonitor tsgm,
      QuiescingExecutors executors,
      OptionsProvider options)
      throws InterruptedException, AbruptExitException {
    getActionEnvFromOptions(options.getOptions(CoreOptions.class));
    PrecomputedValue.REPO_ENV.set(injectable(), new LinkedHashMap<>(repoEnvOption));
    RemoteOptions remoteOptions = options.getOptions(RemoteOptions.class);
    setRemoteExecutionEnabled(remoteOptions != null && remoteOptions.isRemoteExecutionEnabled());
    updateSkyFunctionsSemaphoreSize(options);
    syncPackageLoading(pathPackageLocator, commandId, clientEnv, tsgm, executors, options);

    if (lastAnalysisDiscarded) {
      logger.atInfo().log("Discarding analysis cache because the previous invocation told us to");
      dropConfiguredTargetsNow(eventHandler);
      lastAnalysisDiscarded = false;
    }
    return null;
  }

  /**
   * Updates the size of {@link #oomSensitiveSkyFunctionsSemaphore} based on the provided flag
   * option. If the provided value is 0, remove the semaphore.
   */
  private void updateSkyFunctionsSemaphoreSize(OptionsProvider options) {
    AnalysisOptions analysisOptions = options.getOptions(AnalysisOptions.class);
    if (analysisOptions == null) {
      return;
    }
    Semaphore newSemaphore =
        analysisOptions.oomSensitiveSkyFunctionsSemaphoreSize == 0
            ? null
            : new Semaphore(analysisOptions.oomSensitiveSkyFunctionsSemaphoreSize);
    oomSensitiveSkyFunctionsSemaphore.set(newSemaphore);
  }

  protected void syncPackageLoading(
      PathPackageLocator pathPackageLocator,
      UUID commandId,
      Map<String, String> clientEnv,
      TimestampGranularityMonitor tsgm,
      QuiescingExecutors executors,
      OptionsProvider options)
      throws AbruptExitException {
    PackageOptions packageOptions = options.getOptions(PackageOptions.class);
    try (SilentCloseable c = Profiler.instance().profile("preparePackageLoading")) {
      preparePackageLoading(
          pathPackageLocator,
          packageOptions,
          options.getOptions(BuildLanguageOptions.class),
          commandId,
          clientEnv,
          executors,
          tsgm);
    }
    try (SilentCloseable c = Profiler.instance().profile("setDeletedPackages")) {
      setDeletedPackages(packageOptions.getDeletedPackages());
    }

    incrementalBuildMonitor = new SkyframeIncrementalBuildMonitor();
    invalidateTransientErrors();
    sourceArtifactsSeen.reset();
    outputArtifactsSeen.reset();
    outputArtifactsFromActionCache.reset();
    topLevelArtifactsMetric.reset();
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

  public PathPackageLocator createPackageLocator(
      ExtendedEventHandler eventHandler, List<String> packagePaths, Path workingDirectory) {
    return PathPackageLocator.create(
        directories.getOutputBase(),
        packagePaths,
        eventHandler,
        directories.getWorkspace().asFragment(),
        workingDirectory,
        buildFilesByPriority);
  }

  private CyclesReporter createCyclesReporter() {
    return new CyclesReporter(
        new TargetCycleReporter(packageManager),
        new ActionArtifactCycleReporter(packageManager),
        new TestExpansionCycleReporter(packageManager),
        new RegisteredToolchainsCycleReporter(),
        // TODO(ulfjack): The BzlLoadCycleReporter swallows previously reported cycles
        //  unconditionally! Is that intentional?
        new BzlLoadCycleReporter(),
        new BzlmodRepoCycleReporter());
  }

  public CyclesReporter getCyclesReporter() {
    return cyclesReporter;
  }

  public void setActionExecutionProgressReportingObjects(
      @Nullable ProgressSupplier supplier,
      @Nullable ActionCompletedReceiver completionReceiver,
      @Nullable ActionExecutionStatusReporter statusReporter) {
    skyframeActionExecutor.setActionExecutionProgressReportingObjects(supplier, completionReceiver);
    this.statusReporterRef.set(statusReporter);
  }

  public abstract void detectModifiedOutputFiles(
      ModifiedFileSet modifiedOutputFiles,
      @Nullable Range<Long> lastExecutionTimeRange,
      RemoteArtifactChecker remoteArtifactChecker,
      int fsvcThreads)
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
            ImmutableList.copyOf(targetPatterns), relativeWorkingDirectory);
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
            relativeWorkingDirectory,
            options.compileOneDependency,
            options.buildTestsOnly,
            determineTests,
            ImmutableList.copyOf(options.buildTagFilterList),
            options.buildManualTests,
            options.expandTestSuites,
            TestFilter.forOptions(options));
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
    tryThrowTargetParsingException(eventHandler, targetPatterns, key, evalResult);
    eventHandler.post(new TargetParsingPhaseTimeEvent(timer.stop().elapsed().toMillis()));
    return evalResult.get(key);
  }

  private void tryThrowTargetParsingException(
      ExtendedEventHandler eventHandler,
      List<String> targetPatterns,
      SkyKey key,
      EvaluationResult<TargetPatternPhaseValue> evalResult)
      throws TargetParsingException {
    if (evalResult.hasError()) {
      ErrorInfo errorInfo = evalResult.getError(key);
      TargetParsingException exc;
      if (!errorInfo.getCycleInfo().isEmpty()) {
        exc =
            new TargetParsingException(
                "cycles detected during target parsing", TargetPatterns.Code.CYCLE);
        cyclesReporter.reportCycles(errorInfo.getCycleInfo(), key, eventHandler);
        // Fallback: we don't know which patterns failed, specifically, so we report the entire
        // set as being in error.
        eventHandler.post(PatternExpandingError.failed(targetPatterns, exc.getMessage()));
      } else {
        exc = constructNoCycleTargetParsingException(eventHandler, targetPatterns, errorInfo);
      }
      throw exc;
    }
  }

  private static TargetParsingException constructNoCycleTargetParsingException(
      ExtendedEventHandler eventHandler, List<String> targetPatterns, ErrorInfo errorInfo) {
    Exception e = checkNotNull(errorInfo.getException());
    DetailedExitCode detailedExitCode = traverseExceptionChain(e);
    if (!(e instanceof TargetParsingException)) {
      // If it's a TargetParsingException, then the TargetPatternPhaseFunction has already
      // reported the error, so we don't need to report it again.
      eventHandler.post(PatternExpandingError.failed(targetPatterns, e.getMessage()));
    }

    // Following SkyframeTargetPatternEvaluator, we create with a new TargetParsingException either
    // with an existing DetailedExitCode, or with a FailureDetail Code.
    Throwable cause = e instanceof TargetParsingException ? e.getCause() : e;
    return detailedExitCode != null
        ? new TargetParsingException(e.getMessage(), cause, detailedExitCode)
        : new TargetParsingException(
            e.getMessage(), cause, TargetPatterns.Code.TARGET_PATTERN_PARSE_FAILURE);
  }

  @Nullable
  private static DetailedExitCode traverseExceptionChain(Exception topLevelException) {
    Exception traverseException = topLevelException;
    DetailedExitCode detailedExitCode = null;
    int traverseLevel = 0;
    while (traverseLevel < EXCEPTION_TRAVERSAL_LIMIT) {
      traverseLevel++;
      detailedExitCode = DetailedException.getDetailedExitCode(traverseException);
      if (detailedExitCode != null || traverseException.getCause() == null) {
        break;
      }
      traverseException = (Exception) traverseException.getCause();
    }
    return detailedExitCode;
  }

  public RepositoryMapping getMainRepoMapping(ExtendedEventHandler eventHandler)
      throws InterruptedException, RepositoryMappingResolutionException {
    return getMainRepoMapping(false, DEFAULT_THREAD_COUNT, eventHandler);
  }

  public RepositoryMapping getMainRepoMapping(
      boolean keepGoing, int loadingPhaseThreads, ExtendedEventHandler eventHandler)
      throws InterruptedException, RepositoryMappingResolutionException {
    SkyKey mainRepoMappingKey = RepositoryMappingValue.key(RepositoryName.MAIN);
    EvaluationResult<RepositoryMappingValue> evalResult =
        evaluate(
            ImmutableList.of(mainRepoMappingKey), keepGoing, loadingPhaseThreads, eventHandler);
    if (evalResult.hasError()) {
      ErrorInfo errorInfo = evalResult.getError(mainRepoMappingKey);
      Exception e = errorInfo.getException();
      if (e == null && !errorInfo.getCycleInfo().isEmpty()) {
        cyclesReporter.reportCycles(errorInfo.getCycleInfo(), mainRepoMappingKey, eventHandler);
        throw new RepositoryMappingResolutionException(
            DetailedExitCode.of(
                FailureDetail.newBuilder()
                    .setExternalRepository(
                        FailureDetails.ExternalRepository.newBuilder()
                            .setCode(ExternalRepository.Code.REPOSITORY_MAPPING_RESOLUTION_FAILED)
                            .build())
                    .setMessage("cycles detected during computation of main repo mapping")
                    .build()));
      }
      if (e instanceof DetailedException) {
        throw new RepositoryMappingResolutionException(
            ((DetailedException) e).getDetailedExitCode(), e);
      }
      throw new RepositoryMappingResolutionException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setExternalRepository(FailureDetails.ExternalRepository.getDefaultInstance())
                  .setMessage("unknown error during computation of main repo mapping")
                  .build()),
          e);
    }
    return evalResult.get(mainRepoMappingKey).getRepositoryMapping();
  }

  @Nullable
  private RuleContextConstraintSemantics getRuleContextConstraintSemantics() {
    return ruleContextConstraintSemantics;
  }

  public void setRuleContextConstraintSemantics(
      RuleContextConstraintSemantics ruleContextConstraintSemantics) {
    this.ruleContextConstraintSemantics = ruleContextConstraintSemantics;
  }

  private RegexFilter getExtraActionFilter() {
    return checkNotNull(extraActionFilter);
  }

  private TestTypeResolver getTestTypeResolver() {
    return checkNotNull(testTypeResolver);
  }

  public void setTestTypeResolver(TestTypeResolver testTypeResolver) {
    this.testTypeResolver = testTypeResolver;
  }

  public void setExtraActionFilter(RegexFilter extraActionFilter) {
    this.extraActionFilter = extraActionFilter;
  }

  public void setAndStartWatchdog(ActionExecutionInactivityWatchdog watchdog) {
    this.watchdog = watchdog;
    watchdog.start();
  }

  public AtomicBoolean getIsBuildingExclusiveArtifacts() {
    return isBuildingExclusiveArtifacts;
  }

  /** A progress receiver to track analysis invalidation and update progress messages. */
  protected class SkyframeProgressReceiver implements EvaluationProgressReceiver {
    /**
     * This flag is needed in order to avoid invalidating legacy data when we clear the analysis
     * cache because of --discard_analysis_cache flag. For that case we want to keep the legacy data
     * but get rid of the Skyframe data.
     */
    boolean ignoreInvalidations = false;

    /** This receiver is only needed for execution, so it is null otherwise. */
    @Nullable private EvaluationProgressReceiver executionProgressReceiver = null;

    /**
     * As the ActionLookupValues are marked done in the graph, put it in the map. This map will be
     * "swapped out" for a new one each time it's requested via {@link
     * #getBatchedActionLookupValuesForConflictChecking()}
     */
    @GuardedBy("this")
    private Map<ActionLookupKey, SkyValue> batchedActionLookupValuesForConflictChecking =
        Maps.newConcurrentMap();

    // In non-incremental builds, we want to remove the glob subgraph after the rdep PACKAGE is
    // done. However, edges are not stored. So, we use the `globDeps` map to temporarily store the
    // relationship between GLOB and their dependent GLOBs.
    // After the rdep PACKAGE has been evaluated, all direct or transitive dependent GLOBs will be
    // recursively removed from both the in-memory graph and `globDeps` map.
    private Map<GlobDescriptor, ImmutableList<GlobDescriptor>> globDeps = new ConcurrentHashMap<>();

    @Override
    public void invalidated(SkyKey skyKey, InvalidationState state) {
      if (ignoreInvalidations) {
        return;
      }
      skyframeBuildView.getProgressReceiver().invalidated(skyKey, state);
    }

    @Override
    public void enqueueing(SkyKey skyKey) {
      if (ignoreInvalidations) {
        return;
      }
      skyframeBuildView.getProgressReceiver().enqueueing(skyKey);
      if (executionProgressReceiver != null) {
        executionProgressReceiver.enqueueing(skyKey);
      }
    }

    @Override
    public void stateStarting(SkyKey skyKey, NodeState nodeState) {
      if (NodeState.COMPUTE.equals(nodeState)) {
        skyKeyStateReceiver.computationStarted(skyKey);
      }
    }

    @Override
    public void stateEnding(SkyKey skyKey, NodeState nodeState) {
      if (NodeState.COMPUTE.equals(nodeState)) {
        skyKeyStateReceiver.computationEnded(skyKey);
      }
    }

    @Override
    public void evaluated(
        SkyKey skyKey,
        @Nullable SkyValue newValue,
        @Nullable ErrorInfo newError,
        Supplier<EvaluationSuccessState> evaluationSuccessState,
        EvaluationState state,
        @Nullable GroupedDeps directDeps) {
      if (heuristicallyDropNodes) {
        Object argument = skyKey.argument();
        if (skyKey.functionName().equals(FileValue.FILE)) {
          checkArgument(
              argument instanceof RootedPath,
              "FILE SkyKey (%s) does not have a RootedPath typed argument (%s)",
              skyKey,
              argument);
          memoizingEvaluator.getInMemoryGraph().remove((RootedPath) argument);
        } else if (skyKey.functionName().equals(SkyFunctions.DIRECTORY_LISTING)) {
          checkArgument(
              argument instanceof RootedPath,
              "DIRECTORY_LISTING SkyKey (%s) does not have a RootedPath typed argument (%s)",
              skyKey,
              argument);
          SkyKey directoryListingStateKey = DirectoryListingStateValue.key((RootedPath) argument);
          memoizingEvaluator.getInMemoryGraph().remove(directoryListingStateKey);
        } else if (directDeps != null
            && skyKey.functionName().equals(SkyFunctions.CONFIGURED_TARGET)) {
          maybeDropGenQueryDep(newValue, directDeps);
        }
      }

      if (mergedSkyframeAnalysisExecutionSupplier != null
          && mergedSkyframeAnalysisExecutionSupplier.get()
          && !tracksStateForIncrementality()
          && skyKey instanceof ActionLookupKey
          && newValue != null) {
        synchronized (this) {
          batchedActionLookupValuesForConflictChecking.put((ActionLookupKey) skyKey, newValue);
        }
      }

      if (EvaluationState.BUILT.equals(state)) {
        skyKeyStateReceiver.evaluated(skyKey);
      }
      if (ignoreInvalidations) {
        return;
      }
      skyframeBuildView
          .getProgressReceiver()
          .evaluated(skyKey, newValue, newError, evaluationSuccessState, state, directDeps);
      if (executionProgressReceiver != null) {
        executionProgressReceiver.evaluated(
            skyKey, newValue, newError, evaluationSuccessState, state, directDeps);
      }

      // After a PACKAGE node is evaluated, all targets and the labels associated with this package
      // should have been added to the InMemoryGraph. So it is safe to remove relevant labels from
      // weak interner.
      LabelInterner labelInterner = Label.getLabelInterner();
      if (labelInterner.enabled()
          && skyKey.functionName().equals(SkyFunctions.PACKAGE)
          && newValue != null) {
        checkState(newValue instanceof PackageValue, newValue);

        Package pkg = ((PackageValue) newValue).getPackage();
        Lock writeLock = labelInterner.getLockForLabelTransferToPool(pkg.getPackageIdentifier());
        writeLock.lock();
        try {
          pkg.getTargets()
              .forEach(
                  (name, target) -> {
                    Label label = target.getLabel();
                    labelInterner.removeWeak(label);
                  });
        } finally {
          writeLock.unlock();
        }
      }

      if (!heuristicallyDropNodes || directDeps == null) {
        return;
      }

      if (skyKey.functionName().equals(SkyFunctions.GLOB)) {
        checkState(!globDeps.containsKey(skyKey), skyKey);
        ImmutableList.Builder<GlobDescriptor> directDepGlobsBuilder = ImmutableList.builder();
        for (SkyKey dep : directDeps.getAllElementsAsIterable()) {
          if (dep.functionName().equals(SkyFunctions.GLOB)) {
            checkArgument(dep instanceof GlobDescriptor, dep);
            directDepGlobsBuilder.add((GlobDescriptor) dep);
          }
        }

        ImmutableList<GlobDescriptor> directDepGlobs = directDepGlobsBuilder.build();
        if (!directDepGlobs.isEmpty()) {
          globDeps.put((GlobDescriptor) skyKey, directDepGlobs);
        }
      }

      if (skyKey.functionName().equals(SkyFunctions.PACKAGE)) {
        for (SkyKey dep : directDeps.getAllElementsAsIterable()) {
          if (dep.functionName().equals(SkyFunctions.GLOB)) {
            recursivelyRemoveGlobFromGraph((GlobDescriptor) dep);
          }
        }
      }
    }

    private void maybeDropGenQueryDep(SkyValue newValue, GroupedDeps directDeps) {
      if (!(newValue instanceof RuleConfiguredTargetValue)) {
        return;
      }
      var t = ((RuleConfiguredTargetValue) newValue).getConfiguredTarget();
      if (!t.getRuleClassString().equals("genquery")) {
        return;
      }
      BuildOptions options = t.getConfigurationKey().getOptions();
      if (!options.contains(GenQueryOptions.class)
          || !options.get(GenQueryOptions.class).skipTtvs) {
        return;
      }
      for (SkyKey key : directDeps.getAllElementsAsIterable()) {
        if (key instanceof GenQueryDirectPackageProviderFactory.Key) {
          // The following call can occur several times for the same GENQUERY_SCOPE key in a single
          // Skyframe evaluation, because multiple genquery configured targets may have deps on the
          // same GENQUERY_SCOPE node. It is #removeIfDone and not merely #remove because not-done
          // nodes cannot be removed from the graph, because they may own state which the Skyframe
          // evaluation depends on for its completion, namely, the list of rdeps which must be
          // signaled when the node finishes evaluation.
          memoizingEvaluator.getInMemoryGraph().removeIfDone(key);
          return;
        }
      }
    }

    private void recursivelyRemoveGlobFromGraph(GlobDescriptor root) {
      memoizingEvaluator.getInMemoryGraph().remove(root);
      ImmutableList<GlobDescriptor> adjacentDeps = globDeps.remove(root);
      if (adjacentDeps != null) {
        for (GlobDescriptor nextLevelDep : adjacentDeps) {
          recursivelyRemoveGlobFromGraph(nextLevelDep);
        }
      }
    }

    /**
     * Returns the {@link ActionLookupKey} keyed map of done {@link ActionLookupValue}s since the
     * last time this method was called.
     *
     * <p>Replaces {@link #batchedActionLookupValuesForConflictChecking} with a new empty map.
     *
     * <p>This is only used in skymeld mode AND when we don't keep the incremental state.
     */
    public ImmutableMap<ActionLookupKey, SkyValue>
        getBatchedActionLookupValuesForConflictChecking() {
      checkState(
          checkNotNull(mergedSkyframeAnalysisExecutionSupplier).get()
              && !tracksStateForIncrementality());
      Map<ActionLookupKey, SkyValue> snapshot;
      Map<ActionLookupKey, SkyValue> freshMap = Maps.newConcurrentMap();
      synchronized (this) {
        snapshot = batchedActionLookupValuesForConflictChecking;
        batchedActionLookupValuesForConflictChecking = freshMap;
      }
      return ImmutableMap.copyOf(snapshot);
    }
  }

  public final ExecutionFinishedEvent createExecutionFinishedEvent() {
    return createExecutionFinishedEventInternal()
        .setSourceArtifactsRead(sourceArtifactsSeen.toFilesMetricAndReset())
        .setOutputArtifactsSeen(outputArtifactsSeen.toFilesMetricAndReset())
        .setOutputArtifactsFromActionCache(outputArtifactsFromActionCache.toFilesMetricAndReset())
        .setTopLevelArtifacts(topLevelArtifactsMetric.toFilesMetricAndReset())
        .build();
  }

  @ForOverride
  protected ExecutionFinishedEvent.Builder createExecutionFinishedEventInternal() {
    return ExecutionFinishedEvent.builderWithDefaults();
  }

  final ActionLookupValuesTraversal collectActionLookupValuesInBuild(
      List<ConfiguredTargetKey> topLevelCtKeys, ImmutableSet<AspectKey> aspectKeys)
      throws InterruptedException {
    try (SilentCloseable c =
        Profiler.instance().profile("skyframeExecutor.collectActionLookupValuesInBuild")) {
      ActionLookupValuesTraversal alvTraversal = new ActionLookupValuesTraversal();
      if (!tracksStateForIncrementality()) {
        // If we do not have graph edges, we cannot traverse the graph and find only actions in the
        // current build. In this case we can simply return all ActionLookupValues in the graph,
        // since the graph's lifetime is a single build anyway.
        for (Map.Entry<SkyKey, SkyValue> entry : memoizingEvaluator.getDoneValues().entrySet()) {
          if ((entry.getKey() instanceof ActionLookupKey) && entry.getValue() != null) {
            alvTraversal.accumulate((ActionLookupKey) entry.getKey(), entry.getValue());
          }
        }
        return alvTraversal;
      }

      Map<ActionLookupKey, SkyValue> foundActions =
          new TransitiveActionLookupKeysCollector(
                  SkyframeExecutorWrappingWalkableGraph.of(this),
                  /* globalVisitedSet= */ ImmutableSet.of())
              .collect(Iterables.concat(topLevelCtKeys, aspectKeys));
      foundActions.forEach(alvTraversal::accumulate);
      return alvTraversal;
    }
  }

  /**
   * Collects the transitive ActionLookupKeys of the provided {@code key}, taking into account the
   * set of visited ALKs this build for pruning.
   */
  private ActionLookupValuesCollectionResult collectTransitiveActionLookupValuesOfKey(
      ActionLookupKey key) {
    try (SilentCloseable c =
        Profiler.instance().profile("SkyframeExecutor.collectTransitiveActionLookupValuesOfKey")) {
      if (tracksStateForIncrementality()) {
        var localRef = incrementalTransitiveActionLookupKeysCollector;
        // a null value means that the conflict checker is shut down.
        if (localRef == null) {
          return ActionLookupValuesCollectionResult.empty();
        }
        ImmutableMap<ActionLookupKey, SkyValue> foundTransitiveActionLookupEntities =
            localRef.collect(key);
        return ActionLookupValuesCollectionResult.create(
            foundTransitiveActionLookupEntities.values(),
            ImmutableSet.copyOf(foundTransitiveActionLookupEntities.keySet()));
      }
      // No graph edges available when there's no incrementality. We get the ALVs collected
      // since the last time this method was called.
      ImmutableMap<ActionLookupKey, SkyValue> batchOfActionLookupValues =
          progressReceiver.getBatchedActionLookupValuesForConflictChecking();
      return ActionLookupValuesCollectionResult.create(
          batchOfActionLookupValues.values(), batchOfActionLookupValues.keySet());
    }
  }

  public boolean hasDiffAwareness() {
    return diffAwarenessManager != null;
  }

  /** Uses diff awareness on all the package paths to invalidate changed files. */
  @VisibleForTesting
  public void handleDiffsForTesting(ExtendedEventHandler eventHandler)
      throws InterruptedException, AbruptExitException {
    if (lastAnalysisDiscarded) {
      // Values were cleared last build, but they couldn't be deleted because they were needed for
      // the execution phase. We can delete them now.
      dropConfiguredTargetsNow(eventHandler);
      lastAnalysisDiscarded = false;
    }
    PackageOptions packageOptions = Options.getDefaults(PackageOptions.class);
    packageOptions.checkOutputFiles = false;
    ClassToInstanceMap<OptionsBase> options =
        ImmutableClassToInstanceMap.of(PackageOptions.class, packageOptions);
    handleDiffs(
        eventHandler,
        new OptionsProvider() {
          @Nullable
          @Override
          public <O extends OptionsBase> O getOptions(Class<O> optionsClass) {
            return options.getInstance(optionsClass);
          }

          @Override
          public ImmutableMap<String, Object> getStarlarkOptions() {
            return ImmutableMap.of();
          }

          @Override
          public ImmutableMap<String, Object> getExplicitStarlarkOptions(
              java.util.function.Predicate<? super ParsedOptionDescription> filter) {
            return ImmutableMap.of();
          }
        });
  }

  @CanIgnoreReturnValue
  @Nullable
  protected WorkspaceInfoFromDiff handleDiffs(
      ExtendedEventHandler eventHandler, OptionsProvider options)
      throws InterruptedException, AbruptExitException {
    TimestampGranularityMonitor tsgm = this.tsgm.get();
    modifiedFiles.set(0);
    numSourceFilesCheckedBecauseOfMissingDiffs = 0;

    WorkspaceInfoFromDiff workspaceInfo = null;
    Map<Root, DiffAwarenessManager.ProcessableModifiedFileSet> modifiedFilesByPathEntry =
        Maps.newHashMap();
    Set<Pair<Root, ProcessableModifiedFileSet>> pathEntriesWithoutDiffInformation =
        Sets.newHashSet();
    ImmutableList<Root> pkgRoots = pkgLocator.get().getPathEntries();

    Path workspacePath = directories.getWorkspace();
    EvaluationResult<SkyValue> evaluationResult =
        evaluateSkyKeys(eventHandler, ImmutableList.of(IgnoredPackagePrefixesValue.key()));
    IgnoredPackagePrefixesValue ignoredPackagePrefixesValue =
        (IgnoredPackagePrefixesValue) evaluationResult.get(IgnoredPackagePrefixesValue.key());

    if (diffAwarenessManager != null) {
      for (Root pathEntry : pkgRoots) {
        // Ignored package prefixes are specified relative to the workspace root
        // by definition of .bazelignore. So, we only use ignored paths when the
        // package root is equal to the workspace path.
        ImmutableSet<Path> ignoredPaths = ImmutableSet.of();
        if (workspacePath != null && workspacePath.equals(pathEntry.asPath())) {
          ignoredPaths =
              ignoredPackagePrefixesValue.getPatterns().stream()
                  .map(pathEntry::getRelative)
                  .collect(toImmutableSet());
        }

        DiffAwarenessManager.ProcessableModifiedFileSet modifiedFileSet =
            diffAwarenessManager.getDiff(
                eventHandler, getPathForModifiedFileSet(pathEntry), ignoredPaths, options);
        if (pkgRoots.size() == 1) {
          workspaceInfo = modifiedFileSet.getWorkspaceInfo();
          workspaceInfoFromDiffReceiver.syncWorkspaceInfoFromDiff(
              pathEntry.asPath().asFragment(), workspaceInfo);
        }
        if (modifiedFileSet.getModifiedFileSet().treatEverythingAsModified()) {
          pathEntriesWithoutDiffInformation.add(Pair.of(pathEntry, modifiedFileSet));
        } else {
          modifiedFilesByPathEntry.put(pathEntry, modifiedFileSet);
        }
      }
    }
    BuildRequestOptions buildRequestOptions = options.getOptions(BuildRequestOptions.class);
    int fsvcThreads = buildRequestOptions == null ? 200 : buildRequestOptions.fsvcThreads;
    handleDiffsWithCompleteDiffInformation(tsgm, modifiedFilesByPathEntry, fsvcThreads);
    RepositoryOptions repoOptions = options.getOptions(RepositoryOptions.class);
    handleDiffsWithMissingDiffInformation(
        eventHandler,
        tsgm,
        pathEntriesWithoutDiffInformation,
        options.getOptions(PackageOptions.class).checkOutputFiles,
        repoOptions == null || repoOptions.checkExternalRepositoryFiles,
        fsvcThreads);
    handleClientEnvironmentChanges();
    return workspaceInfo;
  }

  /** Returns the path under which to find the modified file set. */
  @ForOverride
  protected Root getPathForModifiedFileSet(Root root) {
    return root;
  }

  /** Invalidates entries in the client environment. */
  private void handleClientEnvironmentChanges() {
    // Remove deleted client environmental variables.
    ImmutableList<SkyKey> deletedKeys =
        Sets.difference(previousClientEnvironment, clientEnv.get().keySet()).stream()
            .map(ClientEnvironmentFunction::key)
            .collect(toImmutableList());
    recordingDiffer.invalidate(deletedKeys);
    previousClientEnvironment = clientEnv.get().keySet();
    // Inject current client environmental values. We can inject unconditionally without fearing
    // over-invalidation; skyframe will not invalidate an injected key if the key's new value is the
    // same as the old value.
    ImmutableMap.Builder<SkyKey, Delta> newValuesBuilder = ImmutableMap.builder();
    for (Map.Entry<String, String> entry : clientEnv.get().entrySet()) {
      newValuesBuilder.put(
          ClientEnvironmentFunction.key(entry.getKey()),
          Delta.justNew(new ClientEnvironmentValue(entry.getValue())));
    }
    recordingDiffer.inject(newValuesBuilder.buildOrThrow());
  }

  /**
   * Invalidates files under path entries whose corresponding {@link DiffAwareness} gave an exact
   * diff. Removes entries from the given map as they are processed. All of the files need to be
   * invalidated, so the map should be empty upon completion of this function.
   */
  private void handleDiffsWithCompleteDiffInformation(
      TimestampGranularityMonitor tsgm,
      Map<Root, ProcessableModifiedFileSet> modifiedFilesByPathEntry,
      int fsvcThreads)
      throws InterruptedException, AbruptExitException {
    for (Root pathEntry : ImmutableSet.copyOf(modifiedFilesByPathEntry.keySet())) {
      DiffAwarenessManager.ProcessableModifiedFileSet processableModifiedFileSet =
          modifiedFilesByPathEntry.get(pathEntry);
      ModifiedFileSet modifiedFileSet = processableModifiedFileSet.getModifiedFileSet();
      checkState(!modifiedFileSet.treatEverythingAsModified(), pathEntry);
      handleChangedFiles(
          ImmutableList.of(pathEntry),
          getDiff(tsgm, modifiedFileSet, pathEntry, fsvcThreads),
          /* numSourceFilesCheckedIfDiffWasMissing= */ 0);
      processableModifiedFileSet.markProcessed();
    }
  }

  /**
   * Finds and invalidates changed files under path entries whose corresponding {@link
   * DiffAwareness} said all files may have been modified.
   *
   * <p>We need to manually check for changes to known files. This entails finding all dirty file
   * system values under package roots for which we don't have diff information. If at least one
   * path entry doesn't have diff information, then we're going to have to iterate over the skyframe
   * values at least once no matter what.
   */
  protected void handleDiffsWithMissingDiffInformation(
      ExtendedEventHandler eventHandler,
      TimestampGranularityMonitor tsgm,
      Set<Pair<Root, ProcessableModifiedFileSet>> pathEntriesWithoutDiffInformation,
      boolean checkOutputFiles,
      boolean checkExternalRepositoryFiles,
      int fsvcThreads)
      throws InterruptedException {

    ExternalFilesKnowledge externalFilesKnowledge = externalFilesHelper.getExternalFilesKnowledge();
    if (!pathEntriesWithoutDiffInformation.isEmpty()
        || (checkOutputFiles && externalFilesKnowledge.anyOutputFilesSeen)
        || (checkExternalRepositoryFiles && repositoryHelpersHolder != null)
        || (checkExternalRepositoryFiles && externalFilesKnowledge.anyFilesInExternalReposSeen)
        || externalFilesKnowledge.tooManyNonOutputExternalFilesSeen) {
      // We freshly compute knowledge of the presence of external files in the skyframe graph. We
      // use a fresh ExternalFilesHelper instance and only set the real instance's knowledge *after*
      // we are done with the graph scan, lest an interrupt during the graph scan causes us to
      // incorrectly think there are no longer any external files.
      ExternalFilesHelper tmpExternalFilesHelper =
          externalFilesHelper.cloneWithFreshExternalFilesKnowledge();

      invalidateValuesMarkedForInvalidation(eventHandler);

      FilesystemValueChecker fsvc = new FilesystemValueChecker(tsgm, syscallCache, fsvcThreads);

      Set<Root> diffPackageRootsUnderWhichToCheck =
          getDiffPackageRootsUnderWhichToCheck(pathEntriesWithoutDiffInformation);

      EnumSet<FileType> fileTypesToCheck = EnumSet.noneOf(FileType.class);
      Iterable<SkyValueDirtinessChecker> dirtinessCheckers = ImmutableList.of();

      if (!diffPackageRootsUnderWhichToCheck.isEmpty()) {
        dirtinessCheckers =
            Iterables.concat(
                dirtinessCheckers,
                ImmutableList.of(
                    new MissingDiffDirtinessChecker(diffPackageRootsUnderWhichToCheck)));
      }
      if (checkExternalRepositoryFiles && repositoryHelpersHolder != null) {
        dirtinessCheckers =
            Iterables.concat(
                dirtinessCheckers,
                ImmutableList.of(repositoryHelpersHolder.repositoryDirectoryDirtinessChecker()));
      }
      if (checkExternalRepositoryFiles) {
        fileTypesToCheck = EnumSet.of(FileType.EXTERNAL_REPO);
      }
      if (externalFilesKnowledge.tooManyNonOutputExternalFilesSeen
          || !externalFilesKnowledge.nonOutputExternalFilesSeen.isEmpty()) {
        fileTypesToCheck.add(FileType.EXTERNAL);
      }
      // See the comment for FileType.OUTPUT for why we need to consider output files here.
      if (checkOutputFiles) {
        fileTypesToCheck.add(FileType.OUTPUT);
      }
      if (!fileTypesToCheck.isEmpty()) {
        dirtinessCheckers =
            Iterables.concat(
                dirtinessCheckers,
                ImmutableList.of(
                    new ExternalDirtinessChecker(tmpExternalFilesHelper, fileTypesToCheck)));
      }
      checkArgument(!Iterables.isEmpty(dirtinessCheckers));

      logger.atInfo().log(
          "About to scan skyframe graph checking for filesystem nodes of types %s",
          Iterables.toString(fileTypesToCheck));
      ImmutableBatchDirtyResult batchDirtyResult;
      try (SilentCloseable c = Profiler.instance().profile("fsvc.getDirtyKeys")) {
        batchDirtyResult =
            fsvc.getDirtyKeys(
                memoizingEvaluator.getValues(),
                new UnionDirtinessChecker(ImmutableList.copyOf(dirtinessCheckers)));
      }
      handleChangedFiles(
          diffPackageRootsUnderWhichToCheck,
          batchDirtyResult,
          /* numSourceFilesCheckedIfDiffWasMissing= */ batchDirtyResult.getNumKeysChecked());
      // We use the knowledge gained during the graph scan that just completed. Otherwise, naively,
      // once an external file gets into the Skyframe graph, we'll overly-conservatively always
      // think the graph needs to be scanned.
      externalFilesHelper.setExternalFilesKnowledge(
          tmpExternalFilesHelper.getExternalFilesKnowledge());
    } else if (!externalFilesKnowledge.nonOutputExternalFilesSeen.isEmpty()) {
      logger.atInfo().log(
          "About to scan %d external files",
          externalFilesKnowledge.nonOutputExternalFilesSeen.size());
      FilesystemValueChecker fsvc = new FilesystemValueChecker(tsgm, syscallCache, fsvcThreads);
      ImmutableBatchDirtyResult batchDirtyResult;
      try (SilentCloseable c = Profiler.instance().profile("fsvc.getDirtyExternalKeys")) {
        Map<SkyKey, SkyValue> externalDirtyNodes = new ConcurrentHashMap<>();
        for (RootedPath path : externalFilesKnowledge.nonOutputExternalFilesSeen) {
          SkyKey key = FileStateValue.key(path);
          SkyValue value = memoizingEvaluator.getExistingValue(key);
          if (value != null) {
            externalDirtyNodes.put(key, value);
          }
          key = DirectoryListingStateValue.key(path);
          value = memoizingEvaluator.getExistingValue(key);
          if (value != null) {
            externalDirtyNodes.put(key, value);
          }
        }
        batchDirtyResult =
            fsvc.getDirtyKeys(
                externalDirtyNodes,
                new ExternalDirtinessChecker(externalFilesHelper, EnumSet.of(FileType.EXTERNAL)));
      }
      handleChangedFiles(
          ImmutableList.of(), batchDirtyResult, batchDirtyResult.getNumKeysChecked());
    }
    for (Pair<Root, DiffAwarenessManager.ProcessableModifiedFileSet> pair :
        pathEntriesWithoutDiffInformation) {
      pair.getSecond().markProcessed();
    }
  }

  /**
   * Before running the {@link FilesystemValueChecker} ensure that all values marked for
   * invalidation have actually been invalidated (recall that invalidation happens at the beginning
   * of the next evaluate() call), because checking those is a waste of time.
   */
  protected final void invalidateValuesMarkedForInvalidation(ExtendedEventHandler eventHandler)
      throws InterruptedException {
    EvaluationContext evaluationContext =
        newEvaluationContextBuilder()
            .setKeepGoing(false)
            .setParallelism(DEFAULT_THREAD_COUNT)
            .setEventHandler(eventHandler)
            .build();
    memoizingEvaluator.evaluate(ImmutableList.of(), evaluationContext);
  }

  protected final Set<Root> getDiffPackageRootsUnderWhichToCheck(
      Set<Pair<Root, ProcessableModifiedFileSet>> pathEntriesWithoutDiffInformation) {
    Set<Root> diffPackageRootsUnderWhichToCheck = new HashSet<>();
    for (Pair<Root, DiffAwarenessManager.ProcessableModifiedFileSet> pair :
        pathEntriesWithoutDiffInformation) {
      diffPackageRootsUnderWhichToCheck.add(pair.getFirst());
    }
    return diffPackageRootsUnderWhichToCheck;
  }

  protected void handleChangedFiles(
      Collection<Root> diffPackageRootsUnderWhichToCheck,
      Differencer.Diff diff,
      int numSourceFilesCheckedIfDiffWasMissing) {
    int numWithoutNewValues = diff.changedKeysWithoutNewValues().size();
    Iterable<SkyKey> keysToBeChangedLaterInThisBuild = diff.changedKeysWithoutNewValues();
    Map<SkyKey, Delta> changedKeysWithNewValues = diff.changedKeysWithNewValues();

    logDiffInfo(
        diffPackageRootsUnderWhichToCheck,
        keysToBeChangedLaterInThisBuild,
        numWithoutNewValues,
        changedKeysWithNewValues.keySet());

    recordingDiffer.invalidate(keysToBeChangedLaterInThisBuild);
    recordingDiffer.inject(changedKeysWithNewValues);
    modifiedFiles.addAndGet(
        getNumberOfModifiedFiles(keysToBeChangedLaterInThisBuild)
            + getNumberOfModifiedFiles(changedKeysWithNewValues.keySet()));
    numSourceFilesCheckedBecauseOfMissingDiffs += numSourceFilesCheckedIfDiffWasMissing;
    incrementalBuildMonitor.accrue(keysToBeChangedLaterInThisBuild);
    incrementalBuildMonitor.accrue(changedKeysWithNewValues.keySet());
  }

  private static final int MAX_NUMBER_OF_CHANGED_KEYS_TO_LOG = 10;

  private static void logDiffInfo(
      Iterable<Root> pathEntries,
      Iterable<SkyKey> changedWithoutNewValue,
      int numWithoutNewValues,
      Set<SkyKey> changedWithNewValue) {
    int numModified = changedWithNewValue.size() + numWithoutNewValues;
    StringBuilder result =
        new StringBuilder("DiffAwareness found ")
            .append(numModified)
            .append(" modified source files and directory listings");
    if (!Iterables.isEmpty(pathEntries)) {
      result.append(" for ");
      result.append(Joiner.on(", ").join(pathEntries));
    }

    if (numModified > 0) {
      Iterable<SkyKey> allModifiedKeys =
          Iterables.concat(changedWithoutNewValue, changedWithNewValue);
      Iterable<SkyKey> trimmed =
          Iterables.limit(allModifiedKeys, MAX_NUMBER_OF_CHANGED_KEYS_TO_LOG);

      result.append(": ").append(Joiner.on(", ").join(trimmed));

      if (numModified > MAX_NUMBER_OF_CHANGED_KEYS_TO_LOG) {
        result.append(", ...");
      }
    }

    logger.atInfo().log("%s", result);
  }

  private static int getNumberOfModifiedFiles(Iterable<SkyKey> modifiedValues) {
    // We are searching only for changed files, DirectoryListingValues don't depend on
    // child values, that's why they are invalidated separately
    return Iterables.size(
        Iterables.filter(modifiedValues, SkyFunctionName.functionIs(FileStateKey.FILE_STATE)));
  }

  /**
   * A sentinel used in {@link TransitiveActionLookupKeysCollector.VisitActionLookupKey#collected}.
   *
   * <p>Since the traversal is concurrent and {@link ActionLookupKey}s can have many reverse
   * dependencies, it's better to short-circuit before recursively creating a subtask. The presence
   * of this value indicates that another thread already intends to visit the key.
   */
  private static final class ClaimedLookupValueSentinel implements SkyValue {
    private static final ClaimedLookupValueSentinel INSTANCE = new ClaimedLookupValueSentinel();

    private ClaimedLookupValueSentinel() {}
  }

  /**
   * Collects the {@link ActionLookupKey} transitive closure of given {@link ActionLookupKey}s.
   *
   * <p>In the non-Skymeld case, this class is constructed and performs one traversal before
   * shutdown at the end of analysis.
   *
   * <p>In Skymeld, there's a single instance having build lifetime that is reset when the build is
   * done. {@link TransitiveActionLookupKeysCollector#collect} is called multiple times, possibly
   * concurrently, after each top-level {@link ActionLookupKey} finishes its analysis to check for
   * action conflicts.
   */
  private static class TransitiveActionLookupKeysCollector {
    private final WalkableGraph walkableGraph;
    private final Set<ActionLookupKey> globalVisitedSet;

    private TransitiveActionLookupKeysCollector(
        WalkableGraph walkableGraph, Set<ActionLookupKey> globalVisitedSet) {
      this.walkableGraph = walkableGraph;
      this.globalVisitedSet = globalVisitedSet;
    }

    /**
     * Traverses the transitive closure of {@code visitationRoots} and returns an {@link
     * ActionLookupKey} keyed map to corresponding values for all visited keys.
     */
    private Map<ActionLookupKey, SkyValue> collect(Iterable<ActionLookupKey> visitationRoots)
        throws InterruptedException {
      ForkJoinPool executorService =
          NamedForkJoinPool.newNamedPool(
              "find-action-lookup-values-in-build", Runtime.getRuntime().availableProcessors());
      var collected = new ConcurrentHashMap<ActionLookupKey, SkyValue>();
      List<Future<?>> futures = Lists.newArrayListWithCapacity(Iterables.size(visitationRoots));
      for (ActionLookupKey key : visitationRoots) {
        if (tryClaimVisitation(key, collected)) {
          futures.add(executorService.submit(new VisitActionLookupKey(key, collected)));
        }
      }
      try {
        for (Future<?> future : futures) {
          future.get();
        }
        return collected;
      } catch (ExecutionException e) {
        throw new IllegalStateException("Error collecting transitive ActionLookupValues", e);
      } finally {
        if (!executorService.isShutdown() && ExecutorUtil.interruptibleShutdown(executorService)) {
          // Preserve the interrupt status.
          Thread.currentThread().interrupt();
        }
      }
    }

    /**
     * Collects the ALKs in the transitive closure of the given single visitation root which have
     * not been confirmed to be conflict-free.
     *
     * <p>IMPORTANT: due to pruning, the set of returned ActionLookupKey is a SUBSET of the set of
     * elements in the transitive closure of the visitationRoot.
     *
     * <p>This method is the Skymeld implementation. Crucially, this differs from the non-Skymeld
     * implementation in that it is launched from a Skyframe thread, likely belongs to the CPU heavy
     * thread pool, and would queue up recursive tasks in this pool as well. No extra
     * ExecutorService is created here.
     */
    private ImmutableMap<ActionLookupKey, SkyValue> collect(ActionLookupKey visitationRoot) {
      var collected = new ConcurrentHashMap<ActionLookupKey, SkyValue>();
      if (!tryClaimVisitation(visitationRoot, collected)) {
        return ImmutableMap.of();
      }
      // Invoke the recursive task in the same FJP.
      new VisitActionLookupKey(visitationRoot, collected).invoke();
      return ImmutableMap.copyOf(collected);
    }

    /**
     * Attempts to claim ownership of {@code key}'s visitation.
     *
     * @return false if {@code key} is already included in {@link #globalVisitedSet}, was already
     *     claimed or has a value.
     */
    private boolean tryClaimVisitation(
        ActionLookupKey key, ConcurrentHashMap<ActionLookupKey, SkyValue> collected) {
      return !globalVisitedSet.contains(key)
          && collected.putIfAbsent(key, ClaimedLookupValueSentinel.INSTANCE) == null;
    }

    protected final class VisitActionLookupKey extends RecursiveAction {
      private final ActionLookupKey key;
      private final ConcurrentHashMap<ActionLookupKey, SkyValue> collected;

      private VisitActionLookupKey(
          ActionLookupKey key, ConcurrentHashMap<ActionLookupKey, SkyValue> collected) {
        this.key = key;
        this.collected = collected;
      }

      @Override
      public void compute() {
        SkyValue value = null;
        try {
          value = walkableGraph.getValue(key);
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
        }
        if (value == null) { // The value failed to evaluate.
          collected.remove(key);
          return;
        }
        collected.put(key, value);
        Iterable<SkyKey> directDeps;
        try {
          directDeps = walkableGraph.getDirectDeps(key);
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          return;
        }
        var subtasks = new ArrayList<VisitActionLookupKey>();
        for (SkyKey dep : directDeps) {
          if (!(dep instanceof ActionLookupKey)) {
            // The subgraph of dependencies of ActionLookupKeys never has a non-ActionLookupKey
            // depending on an ActionLookupKey. So we can skip any non-ActionLookupKeys in the
            // traversal as an optimization.
            continue;
          }
          ActionLookupKey depKey = (ActionLookupKey) dep;
          if (tryClaimVisitation(depKey, collected)) {
            subtasks.add(new VisitActionLookupKey(depKey, collected));
          }
        }
        invokeAll(subtasks);
      }
    }
  }

  private <T extends SkyValue> EvaluationResult<T> evaluate(
      Iterable<? extends SkyKey> roots,
      boolean keepGoing,
      int numThreads,
      ExtendedEventHandler eventHandler)
      throws InterruptedException {
    EvaluationContext evaluationContext =
        newEvaluationContextBuilder()
            .setKeepGoing(keepGoing)
            .setParallelism(numThreads)
            .setEventHandler(eventHandler)
            .build();
    return memoizingEvaluator.evaluate(roots, evaluationContext);
  }

  private static final UnnecessaryTemporaryStateDropper NULL_UNNECESSARY_TEMPORARY_STATE_DROPPER =
      () -> {};

  private volatile UnnecessaryTemporaryStateDropper dropper =
      NULL_UNNECESSARY_TEMPORARY_STATE_DROPPER;

  private final UnnecessaryTemporaryStateDropperReceiver unnecessaryTemporaryStateDropperReceiver =
      new UnnecessaryTemporaryStateDropperReceiver() {
        @Override
        public void onEvaluationStarted(UnnecessaryTemporaryStateDropper dropper) {
          SkyframeExecutor.this.dropper = dropper;
        }

        @Override
        public void onEvaluationFinished() {
          SkyframeExecutor.this.dropper.drop();
          SkyframeExecutor.this.dropper = NULL_UNNECESSARY_TEMPORARY_STATE_DROPPER;
        }
      };

  protected final EvaluationContext.Builder newEvaluationContextBuilder() {
    return EvaluationContext.newBuilder()
        .setUnnecessaryTemporaryStateDropperReceiver(unnecessaryTemporaryStateDropperReceiver);
  }

  void dropUnnecessaryTemporarySkyframeState() {
    dropper.drop();
  }

  /** Receiver for successfully evaluated/doing computation {@link SkyKey}s. */
  public interface SkyKeyStateReceiver {
    SkyKeyStateReceiver NULL_INSTANCE = new SkyKeyStateReceiver() {};

    /** Called when {@code key}'s associated {@link SkyFunction#compute} is called. */
    default void computationStarted(SkyKey key) {}

    /** Called when {@code key}'s associated {@link SkyFunction#compute} has finished. */
    default void computationEnded(SkyKey key) {}

    /** Called when {@code key} has been evaluated and has a value. */
    default void evaluated(SkyKey key) {}

    default ThreadStateReceiver makeThreadStateReceiver(SkyKey key) {
      return ThreadStateReceiver.NULL_INSTANCE;
    }
  }

  @Nullable
  private Package getExistingPackage(PackageIdentifier id) throws InterruptedException {
    var value = (PackageValue) memoizingEvaluator.getExistingValue(id);
    if (value == null) {
      return null;
    }
    return value.getPackage();
  }

  @VisibleForTesting
  public ConfiguredRuleClassProvider getRuleClassProviderForTesting() {
    return ruleClassProvider;
  }

  @VisibleForTesting
  public PackageSettings getPackageSettingsForTesting() {
    return pkgFactory.getPackageSettingsForTesting();
  }

  @VisibleForTesting
  public BlazeDirectories getBlazeDirectoriesForTesting() {
    return directories;
  }

  @VisibleForTesting
  ActionExecutionStatusReporter getActionExecutionStatusReporterForTesting() {
    return statusReporterRef.get();
  }

  @VisibleForTesting
  public void clearEmittedEventStateForTesting() {
    emittedEventState.clear();
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
      throws InterruptedException, AbruptExitException {
    if (lastAnalysisDiscarded) {
      // Values were cleared last build, but they couldn't be deleted because they were needed for
      // the execution phase. We can delete them now.
      dropConfiguredTargetsNow(eventHandler);
      lastAnalysisDiscarded = false;
    }
    clearSyscallCache();
    invalidateFilesUnderPathForTestingImpl(eventHandler, modifiedFileSet, pathEntry);
  }

  @ForOverride
  protected void invalidateFilesUnderPathForTestingImpl(
      ExtendedEventHandler eventHandler, ModifiedFileSet modifiedFileSet, Root pathEntry)
      throws AbruptExitException, InterruptedException {
    TimestampGranularityMonitor tsgm = this.tsgm.get();
    Differencer.Diff diff;
    if (modifiedFileSet.treatEverythingAsModified()) {
      diff =
          new FilesystemValueChecker(tsgm, syscallCache, /* numThreads= */ 200)
              .getDirtyKeys(
                  memoizingEvaluator.getValues(),
                  DirtinessCheckerUtils.createBasicFilesystemDirtinessChecker());
    } else {
      diff = getDiff(tsgm, modifiedFileSet, pathEntry, /* fsvcThreads= */ 200);
    }
    recordingDiffer.invalidate(diff.changedKeysWithoutNewValues());
    recordingDiffer.inject(diff.changedKeysWithNewValues());
    // Blaze invalidates transient errors on every build.
    invalidateTransientErrors();
  }

  /** Returns a dynamic configuration constructed from the given build options. */
  @VisibleForTesting
  public BuildConfigurationValue getConfigurationForTesting(
      ExtendedEventHandler eventHandler, BuildOptions options)
      throws InterruptedException, OptionsParsingException, InvalidConfigurationException {
    SkyKey key =
        BuildConfigurationKey.withPlatformMapping(
            getPlatformMappingValue(eventHandler, options), options);
    return (BuildConfigurationValue)
        evaluate(
                ImmutableList.of(key),
                /* keepGoing= */ false,
                /* numThreads= */ DEFAULT_THREAD_COUNT,
                eventHandler)
            .get(key);
  }

  /** Returns a particular configured target. */
  @VisibleForTesting
  @Nullable
  public ConfiguredTarget getConfiguredTargetForTesting(
      ExtendedEventHandler eventHandler,
      Label label,
      @Nullable BuildConfigurationValue configuration)
      throws InterruptedException {
    ConfiguredTargetAndData prerequisite =
        getConfiguredTargetAndDataForTesting(eventHandler, label, configuration);
    return prerequisite == null ? null : prerequisite.getConfiguredTarget();
  }

  @VisibleForTesting
  @Nullable
  public ConfiguredTargetAndData getConfiguredTargetAndDataForTesting(
      ExtendedEventHandler eventHandler,
      Label label,
      @Nullable BuildConfigurationValue configuration)
      throws InterruptedException {
    var sink =
        new ConfiguredTargetAndDataProducer.ResultSink() {
          @Nullable private ConfiguredTargetAndData result;

          @Override
          public void acceptConfiguredTargetAndData(ConfiguredTargetAndData value, int index) {
            this.result = value;
          }

          @Override
          public void acceptConfiguredTargetAndDataError(ConfiguredValueCreationException error) {}

          @Override
          public void acceptConfiguredTargetAndDataError(InconsistentNullConfigException error) {}

          @Override
          public void acceptConfiguredTargetAndDataError(NoSuchThingException error) {}
        };

    EvaluationResult<SkyValue> result;
    try (var closer = new EnableAnalysisScope()) {
      result =
          StateMachineEvaluatorForTesting.run(
              new ConfiguredTargetAndDataProducer(
                  ConfiguredTargetKey.builder()
                      .setLabel(label)
                      .setConfiguration(configuration)
                      .build(),
                  /* transitionKeys= */ ImmutableList.of(),
                  TransitiveDependencyState.createForTesting(),
                  sink,
                  /* outputIndex= */ 0),
              memoizingEvaluator,
              getEvaluationContextForTesting(eventHandler));
    }
    if (result != null) {
      try {
        var unused =
            SkyframeErrorProcessor.processAnalysisErrors(
                result,
                cyclesReporter,
                eventHandler,
                /* keepGoing= */ true,
                /* eventBus= */ null,
                bugReporter);
      } catch (ViewCreationFailedException ignored) {
        // Ignored.
      }
    }
    return sink.result;
  }

  private EvaluationContext getEvaluationContextForTesting(ExtendedEventHandler eventHandler) {
    return newEvaluationContextBuilder()
        .setParallelism(DEFAULT_THREAD_COUNT)
        .setEventHandler(eventHandler)
        .build();
  }
}
