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
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.LegacyPackage;
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
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ActionCompletedReceiver;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ProgressSupplier;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.ExitCausingException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.ResourceUsage;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.UnixGlob;
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
import com.google.devtools.build.lib.view.config.InvalidConfigurationException;
import com.google.devtools.build.skyframe.AutoUpdatingGraph;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.InMemoryAutoUpdatingGraph;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeKey;
import com.google.devtools.build.skyframe.NodeProgressReceiver;
import com.google.devtools.build.skyframe.NodeType;
import com.google.devtools.build.skyframe.UpdateResult;

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

import javax.annotation.Nullable;

/**
 * A helper object to support Skyframe-driven execution.
 *
 * <p>This object is mostly used to inject external state, such as the executor engine or
 * some additional artifacts (workspace status and build info artifacts) into NodeBuilders
 * for use during the build.
 */
public final class SkyframeExecutor {
  private AutoUpdatingGraph autoUpdatingGraph;
  private final AutoUpdatingGraph.EmittedEventState emittedEventState =
      new AutoUpdatingGraph.EmittedEventState();
  private final Reporter reporter;
  private final PackageFactory pkgFactory;
  private final WorkspaceStatusAction.Factory workspaceStatusActionFactory;
  private final BlazeDirectories directories;
  @Nullable
  private BatchStat batchStatter;

  // Stores Packages between reruns of the PackageNodeBuilder (because of missing dependencies,
  // within the same update() run) to avoid loading the same package twice (first time loading
  // to find subincludes and declare node dependencies).
  // TODO(bazel-team): remove this cache once we have skyframe-native package loading
  // [skyframe-loading]
  private final ConcurrentMap<String, LegacyPackage> packageNodeBuilderCache =
      Maps.newConcurrentMap();
  private final AtomicInteger numPackagesLoaded = new AtomicInteger(0);

  private SkyframeBuildView skyframeBuildView;
  private ErrorEventListener errorEventListener;
  private ActionLogBufferPathGenerator actionLogBufferPathGenerator;
  private final ImmutableSet<? extends DiffAwareness.Factory> diffAwarenessFactories;
  private Map<Path, DiffAwareness> currentDiffAwarenesses = Maps.newHashMap();

  // AtomicReferences are used here as mutable boxes shared with node builders.
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
  // since it is not yet created when we create the node builders, we have to use a supplier,
  // initialized when the build view is created.
  private final MutableSupplier<ArtifactFactory> artifactFactory = new MutableSupplier<>();
  // Used to give to WriteBuildInfoAction via a supplier. Relying on BuildVariableNode.BUILD_ID
  // would be preferable, but we have no way to have the Action depend on that node directly.
  // Having the BuildInfoNodeBuilder own the supplier is currently not possible either, because then
  // it would be invalidated on every build, since it would depend on the build id node.
  private MutableSupplier<UUID> buildId = new MutableSupplier<>();

  private boolean active = true;
  private boolean lastAnalysisDiscarded = false;
  private final PackageManager packageManager;

  /** Lower limit for size of {@link #allLoadedPackages} to consider clearing CT nodes. */
  private int nodeCacheEvictionLimit = -1;

  /** Union of labels of loaded packages since the last eviction of CT nodes. */
  private Set<PathFragment> allLoadedPackages = ImmutableSet.of();

  /**
   * Upper limit for how many graph versions should dirty nodes be retained for.
   *
   * <p>Specifying a value N means, if the current graph version is V and a node was dirtied (and
   * has remained so) in version U, and U + N &lt;= V, then the node will be marked for deletion
   * and purged in version V+1.
   */
  private long versionWindowForDirtyNodeGc = Long.MAX_VALUE;

  // Use skyframe for execution? Alternative is to use legacy execution codepath.
  // TODO(bazel-team): Remove when legacy codepath is no longer used. [skyframe-execution]
  private final boolean skyframeBuild;

  private final TimestampGranularityMonitor tsgm;

  private final ResourceManager resourceManager;

  /** Used to lock auto-updating graph on legacy calls to get existing nodes. */
  private final Object graphNodeLookupLock = new Object();
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
  private MutableSupplier<BuildConfigurationKey> buildConfigurationKey = new MutableSupplier<>();

  @VisibleForTesting
  public SkyframeExecutor(Reporter reporter, PackageFactory pkgFactory,
                          TimestampGranularityMonitor tsgm, BlazeDirectories directories,
                          WorkspaceStatusAction.Factory workspaceStatusActionFactory,
                          ImmutableList<BuildInfoFactory> buildInfoFactories,
                          Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories) {
    this(reporter, pkgFactory, true, tsgm, directories, workspaceStatusActionFactory,
        buildInfoFactories, diffAwarenessFactories, Predicates.<PathFragment>alwaysFalse());
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
      Predicate<PathFragment> allowedMissingInputs
      ) {
    // Strictly speaking, these arguments are not required for initialization, but all current
    // callsites have them at hand, so we might as well set them during construction.
    this.reporter = Preconditions.checkNotNull(reporter);
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
    this.diffAwarenessFactories = ImmutableSet.copyOf(diffAwarenessFactories);
    this.allowedMissingInputs = allowedMissingInputs;
    resetGraph();
  }

  private ImmutableMap<NodeType, NodeBuilder> nodeBuilders(
      Root buildDataDirectory,
      PackageFactory pkgFactory,
      Predicate<PathFragment> allowedMissingInputs) {
    Map<NodeType, NodeBuilder> map = new HashMap<>();
    map.put(NodeTypes.BUILD_VARIABLE, new BuildVariableNodeBuilder());
    map.put(NodeTypes.FILE_STATE, new FileStateNodeBuilder(tsgm, pkgLocator));
    map.put(NodeTypes.FILE_SYMLINK_CYCLE_UNIQUENESS_NODE,
        new FileSymlinkCycleUniquenessNodeBuilder());
    map.put(NodeTypes.FILE, new FileNodeBuilder(pkgLocator));
    map.put(NodeTypes.DIRECTORY_LISTING, new DirectoryListingNodeBuilder(pkgLocator));
    map.put(NodeTypes.PACKAGE_LOOKUP, new PackageLookupNodeBuilder(pkgLocator, deletedPackages));
    map.put(NodeTypes.CONTAINING_PACKAGE_LOOKUP, new ContainingPackageLookupNodeBuilder());
    map.put(NodeTypes.GLOB, new GlobNodeBuilder());
    map.put(NodeTypes.TARGET_PATTERN, new TargetPatternNodeBuilder(pkgLocator));
    map.put(NodeTypes.RECURSIVE_PKG, new RecursivePkgNodeBuilder());
    map.put(NodeTypes.PACKAGE, new PackageNodeBuilder(
        reporter, pkgFactory, packageManager, showLoadingProgress, packageNodeBuilderCache,
        eventBus, numPackagesLoaded));
    map.put(NodeTypes.TARGET_MARKER, new TargetMarkerNodeBuilder());
    map.put(NodeTypes.TRANSITIVE_TARGET, new TransitiveTargetNodeBuilder());
    map.put(NodeTypes.CONFIGURED_TARGET,
        new ConfiguredTargetNodeBuilder(new BuildViewProvider(), skyframeBuild));
    map.put(NodeTypes.CONFIGURATION_COLLECTION, new ConfigurationCollectionNodeBuilder(
        configurationFactory, buildConfigurationKey, reporter));
    if (skyframeBuild) {
      map.put(NodeTypes.TARGET_COMPLETION,
          new TargetCompletionNodeBuilder(new BuildViewProvider()));
      map.put(NodeTypes.ARTIFACT, new ArtifactNodeBuilder(eventBus, allowedMissingInputs));
      map.put(NodeTypes.BUILD_INFO_COLLECTION, new BuildInfoCollectionNodeBuilder(artifactFactory,
          buildDataDirectory));
      map.put(NodeTypes.BUILD_INFO, new WorkspaceStatusNodeBuilder());
      map.put(NodeTypes.ACTION_EXECUTION,
          new ActionExecutionNodeBuilder(skyframeActionExecutor, tsgm));
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
    autoUpdatingGraph.dump(out);
  }

  public void dumpPackages(PrintStream out) {
    Iterable<NodeKey> packageNodeKeys = Iterables.filter(autoUpdatingGraph.getNodes().keySet(),
        NodeTypes.hasNodeType(NodeTypes.PACKAGE));
    out.println(Iterables.size(packageNodeKeys) + " packages");
    for (NodeKey packageNodeKey : packageNodeKeys) {
      Package pkg = ((PackageNode) autoUpdatingGraph.getNodes().get(packageNodeKey)).getPackage();
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

  class BuildViewProvider {
    /**
     * Returns the current {@link SkyframeBuildView} instance.
     */
    SkyframeBuildView getSkyframeBuildView() {
      return skyframeBuildView;
    }
  }

  /**
   * Reinitializes the Skyframe graph, dropping all previously computed nodes.
   *
   * <p>Be careful with this method as it also deletes all injected nodes. You need to make sure
   * that any necessary build variables are reinjected before the next build. Constants can be put
   * in {@link #reinjectConstantNodes}.
   */
  @ThreadCompatible
  public void resetGraph() {
    emittedEventState.clear();
    progressReceiver = new SkyframeProgressReceiver();
    autoUpdatingGraph = new InMemoryAutoUpdatingGraph(
        nodeBuilders(directories.getBuildDataDirectory(), pkgFactory, allowedMissingInputs),
        progressReceiver, emittedEventState);
    currentDiffAwarenesses.clear();
    if (skyframeBuildView != null) {
      skyframeBuildView.clearLegacyData();
    }
    reinjectConstantNodes();
  }

  /**
   * Nodes whose values are known at startup and guaranteed constant are still wiped from the graph
   * when we create a new one, so they must be re-injected each time we create a new graph.
   */
  private void reinjectConstantNodes() {
    injectBuildInfoFactories();
    needToInjectEmbeddedArtifacts = true;
  }

  /**
   * Deletes all ConfiguredTarget nodes from the Skyframe cache.
   *
   * <p>The next graph update will delete all invalid nodes.
   */
  public void dropConfiguredTargets() {
    if (skyframeBuildView != null) {
      skyframeBuildView.clearInvalidatedConfiguredTargets();
    }
    autoUpdatingGraph.delete(
        // We delete any node that can hold an action -- all subclasses of ActionLookupNode -- as
        // well as ActionExecutionNodes, since they do not depend on ActionLookupNodes.
        NodeType.nodeTypeIsIn(ImmutableSet.of(
            NodeTypes.CONFIGURED_TARGET,
            NodeTypes.ACTION_LOOKUP,
            NodeTypes.BUILD_INFO,
            NodeTypes.TARGET_COMPLETION,
            NodeTypes.BUILD_INFO_COLLECTION,
            NodeTypes.ACTION_EXECUTION)));
  }

  /**
   * Invalidates ConfiguredCollectionNode.
   */
  public void invalidateConfigurationCollection() {
    autoUpdatingGraph.invalidate(ImmutableList.of(ConfigurationCollectionNode.CONFIGURATION_KEY));
  }

  /**
   * Deletes all ConfiguredTarget nodes from the Skyframe cache.
   *
   * <p>After the execution of this method all invalidated and marked for deletion nodes
   * (and the nodes depending on them) will be deleted from the graph.
   *
   * <p>WARNING: Note that a call to this method leaves legacy data inconsistent with Skyframe.
   * The next build should clear the legacy caches.
   */
  private void dropConfiguredTargetsNow() {
    dropConfiguredTargets();
    // Run the invalidator to actually delete the nodes.
    try {
      progressReceiver.ignoreInvalidations = true;
      callUninterruptibly(new Callable<Void>() {
        @Override
        public Void call() throws InterruptedException {
          autoUpdatingGraph.update(ImmutableList.<NodeKey>of(), false,
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
   * Save memory by removing references to configured targets and actions in the Skyframe graph.
   * These nodes must be recreated on subsequent builds. We do not clear the top-level target nodes,
   * since their configured targets are needed for the target completion middleman nodes.
   *
   * <p>The nodes are not deleted during this method call, because they are needed for the execution
   * phase. Instead, their data is cleared. The next build will delete the nodes (and recreate them
   * if necessary).
   */
  private void discardAnalysisCache(Collection<ConfiguredTarget> topLevelTargets) {
    lastAnalysisDiscarded = true;
    for (Map.Entry<NodeKey, Node> entry : autoUpdatingGraph.getNodes().entrySet()) {
      if (!entry.getKey().getNodeType().equals(NodeTypes.CONFIGURED_TARGET)) {
        continue;
      }
      ConfiguredTargetNode ctNode = (ConfiguredTargetNode) entry.getValue();
      if (!topLevelTargets.contains(ctNode.getConfiguredTarget())) {
        ctNode.clear();
      }
    }
  }

  /**
   * Saves memory by clearing analysis objects from Skyframe. If using legacy execution, actually
   * deletes the relevant nodes. If using Skyframe execution, clears their data without deleting
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
    BuildVariableNode.DEFAULTS_PACKAGE_CONTENTS.set(autoUpdatingGraph, defaultsPackageContents);
  }

  /**
   * Injects the top-level artifact options.
   */
  public void injectTopLevelContext(TopLevelArtifactContext options) {
    Preconditions.checkState(skyframeBuild(), "Only inject top-level context in Skyframe full");
    BuildVariableNode.TOP_LEVEL_CONTEXT.set(autoUpdatingGraph, options);
  }

  public void injectWorkspaceStatusData() {
    BuildVariableNode.WORKSPACE_STATUS_KEY.set(autoUpdatingGraph,
        workspaceStatusActionFactory.createWorkspaceStatusAction(
            artifactFactory.get(), WorkspaceStatusNode.ARTIFACT_OWNER, buildId));
  }

  /**
   * Sets the default visibility.
   */
  private void setDefaultVisibility(RuleVisibility defaultVisibility) {
    BuildVariableNode.DEFAULT_VISIBILITY.set(autoUpdatingGraph, defaultVisibility);
  }

  /**
   * Injects the build info factory map that will be used when constructing build info
   * actions/artifacts. Unchanged across the life of the Blaze server, although it must be injected
   * each time the graph is created.
   */
  private void injectBuildInfoFactories() {
    ImmutableMap.Builder<BuildInfoKey, BuildInfoFactory> factoryMapBuilder =
        ImmutableMap.builder();
    for (BuildInfoFactory factory : buildInfoFactories) {
      factoryMapBuilder.put(factory.getKey(), factory);
    }
    BuildVariableNode.BUILD_INFO_FACTORIES.set(autoUpdatingGraph, factoryMapBuilder.build());
  }

  private void setShowLoadingProgress(boolean showLoadingProgressValue) {
    showLoadingProgress.set(showLoadingProgressValue);
  }

  private void setCommandId(UUID commandId) {
    BuildVariableNode.BUILD_ID.set(autoUpdatingGraph, commandId);
    buildId.val = commandId;
  }

  private void invalidateDeletedPackages(Iterable<String> deletedPackages) {
    ArrayList<NodeKey> packagesToInvalidate = Lists.newArrayList();
    for (String deletedPackage : deletedPackages) {
      PathFragment pathFragment = new PathFragment(deletedPackage);
      packagesToInvalidate.add(PackageLookupNode.key(pathFragment));
    }
    autoUpdatingGraph.invalidate(packagesToInvalidate);
  }

  /** Returns the build-info.txt and build-changelist.txt artifacts from the graph. */
  public Collection<Artifact> getWorkspaceStatusArtifacts() throws InterruptedException {
    // Should already be in the graph, unless the user didn't request any targets for analysis.
    UpdateResult<WorkspaceStatusNode> result = autoUpdatingGraph.update(
        ImmutableList.of(WorkspaceStatusNode.NODE_KEY), /*keepGoing=*/false, /*numThreads=*/1,
        reporter);
    WorkspaceStatusNode node = result.get(WorkspaceStatusNode.NODE_KEY);
    return ImmutableList.of(node.getStableArtifact(), node.getVolatileArtifact());
  }

  /**
   * Informs user about number of modified files (source and output files).
   */
  // Note, that number of modified files in some cases can be bigger than actual number of
  // modified files for targets in current request. Skyframe checks for modification all files
  // from previous requests.
  public void informAboutNumberOfModifiedFiles() {
    errorEventListener.info(null, String.format("Found %d modified files", modifiedFiles));
  }

  public Reporter getReporter() {
    return reporter;
  }

  public EventBus getEventBus() {
    return eventBus.get();
  }

  public ImmutableList<Path> getPathEntries() {
    return pkgLocator.get().getPathEntries();
  }

  /**
   * Partitions the given filesystem nodes based on which package path root they are under.
   * Returns a {@link Multimap} {@code m} such that {@code m.containsEntry(k, pe)} is true for
   * each filesystem nodekey {@code k} under a package path root {@code pe}. Note that nodes not
   * under a package path root are not present in the returned {@link Multimap}; these nodes are
   * unconditionally checked for changes on each incremental build.
   */
  private static Multimap<Path, NodeKey> partitionNodeKeysByPackagePathEntry(
      Set<Path> pkgRoots, Iterable<NodeKey> filesystemNodeKeys) {
    ImmutableSetMultimap.Builder<Path, NodeKey> multimapBuilder =
        ImmutableSetMultimap.builder();
    for (NodeKey key : filesystemNodeKeys) {
      Preconditions.checkState(key.getNodeType() == NodeTypes.FILE_STATE
          || key.getNodeType() == NodeTypes.DIRECTORY_LISTING, key);
      Path root = ((RootedPath) key.getNodeName()).getRoot();
      if (pkgRoots.contains(root)) {
        multimapBuilder.put(root, key);
      }
      // We don't need to worry about FileStateNodes for external files because they have a
      // dependency on the build_id and so they get invalidated each build.
    }
    return multimapBuilder.build();
  }

  private Iterable<NodeKey> getNodeKeysPotentiallyAffected(
      Iterable<PathFragment> modifiedSourceFiles, final Path pathEntry) {
    // TODO(bazel-team): change ModifiedFileSet to work with RootedPaths instead of PathFragments.
    Iterable<NodeKey> fileStateNodeKeys = Iterables.transform(modifiedSourceFiles,
        new Function<PathFragment, NodeKey>() {
          @Override
          public NodeKey apply(PathFragment pathFragment) {
            return FileStateNode.key(RootedPath.toRootedPath(pathEntry, pathFragment));
          }
        });
    // TODO(bazel-team): Strictly speaking, we only need to invalidate directory nodes when a file
    // has been created or deleted, not when it has been modified. Unfortunately we
    // do not have that information here, although fancy filesystems could provide it with a
    // hypothetically modified DiffAwareness interface.
    // TODO(bazel-team): Even if we don't have that information, we could avoid invalidating
    // directories when the state of a file does not change by statting them and comparing
    // the new filetype (nonexistent/file/symlink/directory) with the old one.
    Iterable<NodeKey> dirNodeKeys = Iterables.transform(modifiedSourceFiles,
        new Function<PathFragment, NodeKey>() {
          @Override
          public NodeKey apply(PathFragment pathFragment) {
            return DirectoryListingNode.key(RootedPath.toRootedPath(pathEntry,
                pathFragment.getParentDirectory()));
          }
        });
    return Iterables.concat(fileStateNodeKeys, dirNodeKeys);
  }

  private static int getNumberOfModifiedFiles(Iterable<NodeKey> modifiedNodes) {
    // We are searching only for changed files, DirectoryListingNodes don't depend on
    // child nodes, that's why they are invalidated separately
    return Iterables.size(Iterables.filter(modifiedNodes,
        NodeType.nodeTypeIs(NodeTypes.FILE_STATE)));
  }

  /**
   * Uses diff awareness on all the package paths to invalidate changed files.
   */
  @VisibleForTesting
  public void handleDiffs() throws InterruptedException {
    if (lastAnalysisDiscarded) {
      // Nodes were cleared last build, but they couldn't be deleted because they were needed for
      // the execution phase. We can delete them now.
      dropConfiguredTargetsNow();
      lastAnalysisDiscarded = false;
    }
    modifiedFiles = 0;
    Map<Path, ModifiedFileSet> modifiedFilesByPathEntry = Maps.newHashMap();
    Set<Path> pathEntriesWithoutDiffInformation = Sets.newHashSet();
    try {
      for (Path pathEntry : pkgLocator.get().getPathEntries()) {
        DiffAwareness diffAwareness = getDiffAwareness(pathEntry);
        // Note that we must invalidate these files, per the contract of DiffAwareness#getDiff.
        ModifiedFileSet modifiedFileSet = diffAwareness.getDiff();
        if (modifiedFileSet.treatEverythingAsModified()) {
          pathEntriesWithoutDiffInformation.add(pathEntry);
        } else {
          modifiedFilesByPathEntry.put(pathEntry, modifiedFileSet);
        }
      }
      // It's important that this function *not* exit before here, lest we forget to invalidate the
      // changed files.
      handleDiffsWithCompleteDiffInformation(modifiedFilesByPathEntry);
      handleDiffsWithMissingDiffInformation(pathEntriesWithoutDiffInformation);
    } finally {
      // Protect against early termination before processing the diffs.
      Preconditions.checkState(modifiedFilesByPathEntry.isEmpty());
    }
  }

  /**
   * Invalidates files under path entries whose corresponding {@link DiffAwareness} gave an exact
   * diff. Removes entries from the given map as they are processed. All of the files need to be
   * invalidated, so the map should be empty upon completion of this function.
   */
  private void handleDiffsWithCompleteDiffInformation(
      Map<Path, ModifiedFileSet> modifiedFilesByPathEntry) {
    // It's important that the below code be uninterruptible, since we already promised to
    // invalidate these files.
    for (Path pathEntry : ImmutableSet.copyOf(modifiedFilesByPathEntry.keySet())) {
      ModifiedFileSet modifiedFileSet = modifiedFilesByPathEntry.get(pathEntry);
      Preconditions.checkState(!modifiedFileSet.treatEverythingAsModified(), pathEntry);
      Iterable<NodeKey> dirtyNodes = getNodeKeysPotentiallyAffected(
          modifiedFileSet.modifiedSourceFiles(), pathEntry);
      modifiedFiles += getNumberOfModifiedFiles(dirtyNodes);
      invalidateAndAccrueChangedFiles(dirtyNodes);
      modifiedFilesByPathEntry.remove(pathEntry);
    }
  }

  /**
   * Finds and invalidates changed files under path entries whose corresponding
   * {@link DiffAwareness} said all files may have been modified.
   */
  private void handleDiffsWithMissingDiffInformation(Set<Path> pathEntriesWithoutDiffInformation)
      throws InterruptedException {
    // Before running the FilesystemNodeChecker, ensure that all nodes marked for invalidation
    // have actually been invalidated (recall that invalidation happens at the beginning of the
    // next update call), because checking those is a waste of time.
    autoUpdatingGraph.update(ImmutableList.<NodeKey>of(), false,
        AutoUpdatingGraph.DEFAULT_THREAD_COUNT, reporter);
    FilesystemNodeChecker fsnc = new FilesystemNodeChecker(autoUpdatingGraph, tsgm);
    // We need to manually check for changes to known files. This entails finding all dirty file
    // system nodes under package roots for which we don't have diff information. If at least
    // one path entry doesn't have diff information, then we're going to have to iterate over
    // the skyframe graph at least once no matter what so we might as well do so now and avoid
    // doing so more than once.
    Iterable<NodeKey> filesystemNodeKeys = fsnc.getFilesystemNodeKeys();
    // Partition by package path entry.
    Multimap<Path, NodeKey> nodeKeysByPathEntry = partitionNodeKeysByPackagePathEntry(
        ImmutableSet.copyOf(pkgLocator.get().getPathEntries()), filesystemNodeKeys);
    // Contains all file system nodes that we need to check for dirtiness.
    List<Iterable<NodeKey>> nodesToCheckManually = Lists.newArrayList();
    for (Path pathEntry : pathEntriesWithoutDiffInformation) {
      nodesToCheckManually.add(nodeKeysByPathEntry.get(pathEntry));
    }
    try {
      Collection<NodeKey> dirtyNodes =
          fsnc.getDirtyFilesystemNodes(Iterables.concat(nodesToCheckManually));
      modifiedFiles += getNumberOfModifiedFiles(dirtyNodes);
      invalidateAndAccrueChangedFiles(dirtyNodes);
    } catch (InterruptedException e) {
      for (Path pathEntry : pathEntriesWithoutDiffInformation) {
        // The diffs from these paths haven't been processed fully, so we need to reset the
        // diff awareness strategies, per {@link DiffAwareness#getDiff}.
        currentDiffAwarenesses.remove(pathEntry);
      }
      throw e;
    }
  }

  private void invalidateAndAccrueChangedFiles(Iterable<NodeKey> nodes) {
    autoUpdatingGraph.invalidate(nodes);
    if (skyframeBuild()) {
      incrementalBuildMonitor.accrue(nodes);
    }
  }

  /**
   * Returns the {@link DiffAwareness} to use for finding changes to files under the given path.
   * This will either be an old diff awareness for the path that is still good per
   * {@link DiffAwareness#canStillBeUsed}, or a fresh one.
   */
  @VisibleForTesting
  public DiffAwareness getDiffAwareness(Path pathEntry) {
    DiffAwareness currentDiffAwareness = currentDiffAwarenesses.get(pathEntry);
    if (currentDiffAwareness != null && currentDiffAwareness.canStillBeUsed()) {
      return currentDiffAwareness;
    }
    DiffAwareness newDiffAwareness = null;
    for (DiffAwareness.Factory factory : diffAwarenessFactories) {
      newDiffAwareness = factory.maybeCreate(pathEntry, pkgLocator.get().getPathEntries());
      if (newDiffAwareness != null) {
        break;
      }
    }
    if (newDiffAwareness == null) {
      newDiffAwareness = new BlindDiffAwareness();
    }
    currentDiffAwarenesses.put(pathEntry, newDiffAwareness);
    return newDiffAwareness;
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
   * Prepares the graph for loading.
   *
   * <p>MUST be run before every incremental build.
   */
  @VisibleForTesting  // productionVisibility = Visibility.PRIVATE
  public void preparePackageLoading(PathPackageLocator pkgLocator, RuleVisibility defaultVisibility,
      boolean showLoadingProgress,
      Preprocessor.Factory preprocessorFactory,
      String defaultsPackageContents, UUID commandId) {
    Preconditions.checkNotNull(pkgLocator);
    setActive(true);

    setCommandId(commandId);
    setShowLoadingProgress(showLoadingProgress);
    setDefaultVisibility(defaultVisibility);
    setupDefaultPackage(defaultsPackageContents);
    setPackageLocator(pkgLocator);

    syscalls.set(new PerBuildSyscallCache());
    if (preprocessorFactory != null) {
      pkgFactory.setPreprocessor(preprocessorFactory.newPreprocessor(getPackageManager()));
    }
    emittedEventState.clear();

    // If the PackageNodeBuilder was interrupted, there may be stale entries here.
    packageNodeBuilderCache.clear();
    numPackagesLoaded.set(0);

    // Reset the stateful SkyframeCycleReporter, which contains cycles from last run.
    cyclesReporter.set(createCyclesReporter());
  }

  /**
   * The node types whose builders have direct access to the package locator. They need to be
   * invalidated if the package locator changes.
   */
  private static final Set<NodeType> PACKAGE_LOCATOR_DEPENDENT_NODES =
      ImmutableSet.of(NodeTypes.FILE_STATE, NodeTypes.FILE, NodeTypes.DIRECTORY_LISTING,
          NodeTypes.PACKAGE_LOOKUP, NodeTypes.TARGET_PATTERN);

  @SuppressWarnings("unchecked")
  private void setPackageLocator(PathPackageLocator pkgLocator) {
    PathPackageLocator oldLocator = this.pkgLocator.getAndSet(pkgLocator);
    if ((oldLocator == null || !oldLocator.getPathEntries().equals(pkgLocator.getPathEntries()))) {
      autoUpdatingGraph.delete(NodeType.nodeTypeIsIn(PACKAGE_LOCATOR_DEPENDENT_NODES));

      // The package path is read not only by NodeBuilders but also by some other code paths.
      // We need to take additional steps to keep the corresponding data structures in sync.
      // (Some of the additional steps are carried out by ConfiguredTargetNodeInvalidationListener,
      // and some by BuildView#buildHasIncompatiblePackageRoots and #updateSkyframe.)
      currentDiffAwarenesses.clear();
    }
  }

  /**
   * Specifies the current {@link SkyframeBuildView} instance. This should only be set once over the
   * lifetime of the Blaze server, except in tests.
   */
  public void setSkyframeBuildView(SkyframeBuildView skyframeBuildView) {
    this.skyframeBuildView = skyframeBuildView;
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
   * Sets the listener to use for reporting errors.
   */
  public void setErrorEventListener(ErrorEventListener listener) {
    this.errorEventListener = listener;
  }

  /**
   * Sets the path for action log buffers.
   */
  public void setActionOutputRoot(Path actionOutputRoot) {
    Preconditions.checkNotNull(actionOutputRoot);
    this.actionLogBufferPathGenerator = new ActionLogBufferPathGenerator(actionOutputRoot);
    this.skyframeActionExecutor.setActionLogBufferPathGenerator(actionLogBufferPathGenerator);
  }

  /**
   * Asks the Skyframe evaluator to build the node for BuildConfigurationCollection and
   * returns result. Also invalidates {@link BuildVariableNode#BUILD_OPTIONS},
   * {@link BuildVariableNode#TEST_ENVIRONMENT_VARIABLES} and
   * {@link BuildVariableNode#BLAZE_DIRECTORIES} if they have changed.
   */
  public BuildConfigurationCollection createConfigurations(boolean keepGoing,
      ConfigurationFactory configurationFactory, BuildConfigurationKey configurationKey)
      throws InvalidConfigurationException, InterruptedException {

    this.buildConfigurationKey.val = configurationKey;
    this.configurationFactory.val = configurationFactory;
    BuildOptions buildOptions = configurationKey.getBuildOptions();
    Map<String, String> testEnv = BuildConfiguration.getTestEnv(
        buildOptions.get(BuildConfiguration.Options.class).testEnvironment,
        configurationKey.getClientEnv());
    // TODO(bazel-team): find a way to use only BuildConfigurationKey instead of BuildOptions,
    // TestEnvironmentVariables and BlazeDirectories. There is a problem only with
    // TestEnvironmentVariables because BuildConfigurationKey stores client environment variables
    // and we don't want to rebuild everything when any variable changes.
    BuildVariableNode.BUILD_OPTIONS.set(autoUpdatingGraph, buildOptions);
    BuildVariableNode.TEST_ENVIRONMENT_VARIABLES.set(autoUpdatingGraph, testEnv);
    BuildVariableNode.BLAZE_DIRECTORIES.set(autoUpdatingGraph, configurationKey.getDirectories());

    UpdateResult<ConfigurationCollectionNode> result =
        autoUpdatingGraph.update(Arrays.asList(ConfigurationCollectionNode.CONFIGURATION_KEY),
        keepGoing, AutoUpdatingGraph.DEFAULT_THREAD_COUNT, errorEventListener);
    if (result.hasError()) {
      Throwable e = result.getError(ConfigurationCollectionNode.CONFIGURATION_KEY).getException();
      Throwables.propagateIfInstanceOf(e, InvalidConfigurationException.class);
      throw new IllegalStateException(
          "Unknown error during ConfigurationCollectionNode evaluation", e);
    }
    Preconditions.checkState(result.values().size() == 1,
        "Result of update must contain exactly one value " + result);
    return Iterables.getOnlyElement(result.values()).getConfigurationCollection();
  }

  private Iterable<ActionLookupNode> getActionLookupNodes() {
    // This filter keeps subclasses of ActionLookupNode.
    return Iterables.filter(autoUpdatingGraph.getDoneNodes().values(), ActionLookupNode.class);
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
      skyframeActionExecutor.findAndStoreArtifactConflicts(getActionLookupNodes());
      skyframeBuildView.resetEvaluatedConfiguredTargetFlag();
      // The invalidated configured targets flag will be reset later in the update call.
    }
    return skyframeActionExecutor.badActions();
  }

  /**
   * Asks the Skyframe evaluator to build the nodes corresponding to the given artifacts.
   *
   * <p>The returned artifacts should be built and present on the filesystem after the call
   * completes.
   */
  public UpdateResult<ArtifactNode> buildArtifacts(
      Executor executor,
      Set<Artifact> artifacts,
      boolean keepGoing,
      int numJobs,
      ActionCacheChecker actionCacheChecker,
      @Nullable NodeProgressReceiver executionProgressReceiver) throws InterruptedException {
    checkActive();
    Preconditions.checkState(actionLogBufferPathGenerator != null);

    skyframeActionExecutor.prepareForExecution(executor, keepGoing, actionCacheChecker);

    resourceManager.resetResourceUsage();
    try {
      progressReceiver.executionProgressReceiver = executionProgressReceiver;
      return autoUpdatingGraph.update(ArtifactNode.mandatoryKeys(artifacts), keepGoing, numJobs,
          errorEventListener);
    } finally {
      progressReceiver.executionProgressReceiver = null;
      // Also releases thread locks.
      resourceManager.resetResourceUsage();
    }
  }

  UpdateResult<TargetPatternNode> targetPatterns(Iterable<NodeKey> patternNodeKeys,
      boolean keepGoing, ErrorEventListener listener) throws InterruptedException {
    checkActive();
    return autoUpdatingGraph.update(patternNodeKeys, keepGoing,
        AutoUpdatingGraph.DEFAULT_THREAD_COUNT, listener);
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
    final Collection<NodeKey> nodeKeys = ConfiguredTargetNode.keys(lacs);
    UpdateResult<Node> result;
    try {
      result = callUninterruptibly(new Callable<UpdateResult<Node>>() {
        @Override
        public UpdateResult<Node> call() throws Exception {
          synchronized (graphNodeLookupLock) {
            try {
              skyframeBuildView.enableAnalysis(true);
              return autoUpdatingGraph.update(nodeKeys, false,
                  AutoUpdatingGraph.DEFAULT_THREAD_COUNT, errorEventListener);
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
    for (Node value : result.values()) {
      ConfiguredTargetNode ctNode = (ConfiguredTargetNode) value;
      cts.add(ctNode.getConfiguredTarget());
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
    if (autoUpdatingGraph.getExistingNodeForTesting(
        BuildVariableNode.WORKSPACE_STATUS_KEY.getKeyForTesting()) == null) {
      injectWorkspaceStatusData();
    }
    return Iterables.getFirst(getConfiguredTargets(ImmutableList.of(
        new LabelAndConfiguration(label, configuration))), null);
  }

  /**
   * Invalidates Skyframe nodes corresponding to the given set of modified files under the given
   * path entry.
   *
   * <p>May throw an {@link InterruptedException}, which means that no nodes have been invalidated.
   */
  @VisibleForTesting
  public void invalidateFilesUnderPathForTesting(ModifiedFileSet modifiedFileSet, Path pathEntry)
      throws InterruptedException {
    if (lastAnalysisDiscarded) {
      // Nodes were cleared last build, but they couldn't be deleted because they were needed for
      // the execution phase. We can delete them now.
      dropConfiguredTargetsNow();
      lastAnalysisDiscarded = false;
    }
    Iterable<NodeKey> keys;
    if (modifiedFileSet.treatEverythingAsModified()) {
      keys = new FilesystemNodeChecker(autoUpdatingGraph, tsgm).getDirtyFilesystemNodeKeys();
    } else {
      keys = getNodeKeysPotentiallyAffected(modifiedFileSet.modifiedSourceFiles(), pathEntry);
    }
    syscalls.set(new PerBuildSyscallCache());
    autoUpdatingGraph.invalidate(keys);
    // Blaze invalidates (transient) errors on every build.
    invalidateErrors();
  }

  /**
   * Invalidates SkyFrame nodes that may have failed for transient reasons.
   */
  @VisibleForTesting  // productionVisibility = Visibility.PRIVATE
  public void invalidateErrors() {
    checkActive();
    autoUpdatingGraph.invalidateErrors();
  }

  @VisibleForTesting
  public TimestampGranularityMonitor getTimestampGranularityMonitorForTesting() {
    return tsgm;
  }

  /**
   * Configures a given set of configured targets.
   */
  public UpdateResult<ConfiguredTargetNode> configureTargets(List<LabelAndConfiguration> nodes,
      boolean keepGoing) throws InterruptedException {
    checkActive();
    List<NodeKey> nodeNames = Lists.newArrayListWithCapacity(nodes.size());
    for (LabelAndConfiguration node : nodes) {
      nodeNames.add(ConfiguredTargetNode.key(node));
    }

    // Make sure to not run too many analysis threads. This can cause memory thrashing.
    return autoUpdatingGraph.update(nodeNames, keepGoing,
        ResourceUsage.getAvailableProcessors(), errorEventListener);
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
     * Loads the specified {@link TransitiveTargetNode}s.
     */
    UpdateResult<TransitiveTargetNode> loadTransitiveTargets(
        Iterable<Target> targetsToVisit, Iterable<Label> labelsToVisit, boolean keepGoing)
        throws InterruptedException {
      List<NodeKey> nodeNames = new ArrayList<>();
      for (Target target : targetsToVisit) {
        nodeNames.add(TransitiveTargetNode.key(target.getLabel()));
      }
      for (Label label : labelsToVisit) {
        nodeNames.add(TransitiveTargetNode.key(label));
      }

      return autoUpdatingGraph.update(nodeNames, keepGoing, AutoUpdatingGraph.DEFAULT_THREAD_COUNT,
          errorEventListener);
    }

    public Set<Package> retrievePackages(Set<PathFragment> packageNames) {
      final List<NodeKey> nodeNames = new ArrayList<>();
      for (PathFragment pkgName : packageNames) {
        nodeNames.add(PackageNode.key(pkgName));
      }

      try {
        return callUninterruptibly(new Callable<Set<Package>>() {
          @Override
          public Set<Package> call() throws Exception {
            UpdateResult<PackageNode> result = autoUpdatingGraph.update(nodeNames, false,
                ResourceUsage.getAvailableProcessors(), errorEventListener);
            Preconditions.checkState(!result.hasError(),
                "unexpected errors: %s", result.errorMap());
            Set<Package> packages = Sets.newHashSet();
            for (PackageNode node : result.values()) {
              packages.add(node.getPackage());
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
          Preconditions.checkState(artifactOwner instanceof ActionLookupNode.ActionLookupKey,
              "", artifact, artifactOwner);
          NodeKey actionLookupKey =
              ActionLookupNode.key((ActionLookupNode.ActionLookupKey) artifactOwner);

          synchronized (graphNodeLookupLock) {
            // Note that this will crash (attempting to run a configured target node builder after
            // analysis) after a failed --nokeep_going analysis in which the configured target that
            // failed was a (transitive) dependency of the configured target that should generate
            // this action. We don't expect callers to query generating actions in such cases.
            UpdateResult<ActionLookupNode> result = autoUpdatingGraph.update(
                ImmutableList.of(actionLookupKey), false,
                ResourceUsage.getAvailableProcessors(), errorEventListener);
            return result.hasError()
                ? null
                : result.get(actionLookupKey).getGeneratingAction(artifact);
          }
        }
      });
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  public PackageManager getPackageManager() {
    return packageManager;
  }

  class SkyframePackageLoader {
    /**
     * Looks up a particular package (used after the loading phase).
     *
     * <p>Note that this method needs to be synchronized since InMemoryAutoUpdatingGraph.update()
     * method does not support concurrent calls.
     */
    Package getPackage(ErrorEventListener listener, String pkgName) throws InterruptedException,
        NoSuchPackageException {
      synchronized (graphNodeLookupLock) {
        NodeKey key = PackageNode.key(new PathFragment(pkgName));
        UpdateResult<PackageNode> result =
            autoUpdatingGraph.update(ImmutableList.of(key), false,
                AutoUpdatingGraph.DEFAULT_THREAD_COUNT, listener);
        if (result.hasError()) {
          if (!Iterables.isEmpty(result.getError().getCycleInfo())) {
            reportCycles(result.getError().getCycleInfo(), key);
            // This can only happen if a package is freshly loaded outside of the target parsing
            // or loading phase
            throw new BuildFileContainsErrorsException(pkgName,
                "Cycle encountered while loading package " + pkgName);
          }
          Throwable e = result.getError().getException();
          // PackageNodeBuilder should be catching, swallowing, and rethrowing all transitive
          // errors as NoSuchPackageExceptions.
          Throwables.propagateIfInstanceOf(e, NoSuchPackageException.class);
          throw new IllegalStateException("Unexpected Exception type from PackageNode.", e);
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
      packageNodeBuilderCache.clear();
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
  public AutoUpdatingGraph getGraphForTesting() {
    return autoUpdatingGraph;
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
   * Stores the set of loaded packages and, if needed, evicts ConfiguredTarget nodes.
   *
   * <p>The set represents all packages from the transitive closure of the top-level targets from
   * the latest build.
   */
  @ThreadCompatible
  public void updateLoadedPackageSet(Set<PathFragment> loadedPackages) {
    Preconditions.checkState(nodeCacheEvictionLimit >= 0,
        "should have called setMinLoadedPkgCountForCtNodeEviction earlier");

    // Make a copy to avoid nesting SetView objects. It also computes size(), which we need below.
    Set<PathFragment> union = ImmutableSet.copyOf(Sets.union(allLoadedPackages, loadedPackages));

    if (union.size() < nodeCacheEvictionLimit ||
        isBuildSubsetOrSupersetOfPreviousBuild(allLoadedPackages, loadedPackages)) {
      allLoadedPackages = union;
    } else {
      dropConfiguredTargets();
      allLoadedPackages = loadedPackages;
    }
  }

  public void sync(Preprocessor.Factory preprocessorFactory,
      PackageCacheOptions packageCacheOptions, Path workingDirectory,
      String defaultsPackageContents, UUID commandId) throws InterruptedException {
    PathPackageLocator packageLocator = PathPackageLocator.create(
        packageCacheOptions.packagePath, getReporter(), directories.getWorkspace(),
        workingDirectory);

    preparePackageLoading(packageLocator,
        packageCacheOptions.defaultVisibility, packageCacheOptions.showLoadingProgress,
        preprocessorFactory, defaultsPackageContents, commandId);
    setDeletedPackages(ImmutableSet.copyOf(packageCacheOptions.deletedPackages));
    this.nodeCacheEvictionLimit = packageCacheOptions.minLoadedPkgCountForCtNodeEviction;
    setSkylarkEnabled(packageCacheOptions);

    incrementalBuildMonitor = new SkyframeIncrementalBuildMonitor();
    invalidateErrors();
    handleDiffs();
  }

  public void setSkylarkEnabled(PackageCacheOptions packageCacheOptions) {
    pkgFactory.setSkylarkEnabled(packageCacheOptions.enableSkylark);
  }

  private CyclesReporter createCyclesReporter() {
    return new CyclesReporter(new TransitiveTargetCycleReporter(packageManager),
        new ConfiguredTargetCycleReporter(packageManager));
  }

  CyclesReporter getCyclesReporter() {
    return cyclesReporter.get();
  }

  /** Convenience method with same semantics as {@link CyclesReporter#reportCycles}. */
  public void reportCycles(Iterable<CycleInfo> cycles, NodeKey topLevelKey) {
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

  public void prepareExecution() throws ExitCausingException, InterruptedException {
    Preconditions.checkState(skyframeBuild(),
        "Cannot prepare execution phase if not using Skyframe full");
    maybeInjectEmbeddedArtifacts();

    // Detect external modifications in the output tree.
    FilesystemNodeChecker fsnc = new FilesystemNodeChecker(autoUpdatingGraph, tsgm);
    autoUpdatingGraph.invalidate(fsnc.getDirtyActionNodes(batchStatter));
    modifiedFiles += fsnc.getNumberOfModifiedOutputFiles();
  }

  @VisibleForTesting void maybeInjectEmbeddedArtifacts() throws ExitCausingException {
    // The blaze client already ensures that the contents of the embedded binaries never change,
    // so we just need to make sure that the appropriate artifacts are present in the skyframe
    // graph.

    if (!needToInjectEmbeddedArtifacts) {
      return;
    }

    Preconditions.checkNotNull(artifactFactory.get());
    Preconditions.checkNotNull(binTools);
    Map<NodeKey, Node> nodes = Maps.newHashMap();
    // Blaze separately handles the symlinks that target these binaries. See BinTools#setupTool.
    for (Artifact artifact : binTools.getAllEmbeddedArtifacts(artifactFactory.get())) {
      FileArtifactNode fileArtifactNode = null;
      try {
        fileArtifactNode = FileArtifactNode.create(artifact);
      } catch (IOException e) {
        // See ExtractData in blaze.cc.
        String message = "Error: corrupt installation: file " + artifact.getPath() + " missing. "
            + "Please remove '" + directories.getInstallBase() + "' and try again.";
        throw new ExitCausingException(message, ExitCode.LOCAL_ENVIRONMENTAL_ERROR, e);
      }
      nodes.put(ArtifactNode.key(artifact, /*isMandatory=*/true), fileArtifactNode);
    }
    autoUpdatingGraph.inject(nodes);
    needToInjectEmbeddedArtifacts = false;
  }

  /**
   * Sets the upper limit for the number of graph versions that dirty nodes may be retained for.
   *
   * <p>Specifying a value N means, if the current graph version is V and a node was dirtied (and
   * has remained so) in version U, and U + N &lt;= V, then the node will be marked for deletion
   * and purged in version V + 1.
   *
   * @param versionWindow a non-negative number indicating the length of the window
   */
  public void setVersionWindowForDirtyNodeGc(long versionWindow) {
    Preconditions.checkArgument(versionWindow >= 0);
    this.versionWindowForDirtyNodeGc = versionWindow;
  }

  /**
   * Mark dirty nodes for deletion if they've been dirty for longer than
   * {@link #versionWindowForDirtyNodeGc} versions.
   */
  void deleteOldNodes() {
    // TODO(bazel-team): perhaps we should come up with a separate GC class dedicated to maintaining
    // node garbage. If we ever do so, this logic should be moved there.
    autoUpdatingGraph.deleteDirty(versionWindowForDirtyNodeGc);
  }

  private class SkyframeProgressReceiver implements NodeProgressReceiver {

    /**
     * This flag is needed in order to avoid invalidating legacy data when we clear the
     * analysis cache because of --discard_analysis_cache flag. For that case we want to keep
     * the legacy data but get rid of the Skyframe data.
     */
    private boolean ignoreInvalidations = false;
    /** This receiver is only needed for execution, so it is null otherwise. */
    @Nullable NodeProgressReceiver executionProgressReceiver = null;

    @Override
    public void invalidated(Node node, InvalidationState state) {
      if (ignoreInvalidations) {
        return;
      }
      if (skyframeBuildView != null) {
        skyframeBuildView.getInvalidationReceiver().invalidated(node, state);
      }
    }

    @Override
    public void enqueueing(NodeKey nodeKey) {
      if (ignoreInvalidations) {
        return;
      }
      if (skyframeBuildView != null) {
        skyframeBuildView.getInvalidationReceiver().enqueueing(nodeKey);
      }
      if (executionProgressReceiver != null) {
        executionProgressReceiver.enqueueing(nodeKey);
      }
    }

    @Override
    public void evaluated(NodeKey nodeKey, Node node, EvaluationState state) {
      if (ignoreInvalidations) {
        return;
      }
      if (skyframeBuildView != null) {
        skyframeBuildView.getInvalidationReceiver().evaluated(nodeKey, node, state);
      }
      if (executionProgressReceiver != null) {
        executionProgressReceiver.evaluated(nodeKey, node, state);
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
