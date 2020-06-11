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
package com.google.devtools.build.lib.query2.query;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.TargetEdgeObserver;
import com.google.devtools.build.lib.pkgcache.TargetPatternPreloader;
import com.google.devtools.build.lib.pkgcache.TargetProvider;
import com.google.devtools.build.lib.pkgcache.TransitivePackageLoader;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.compat.FakeLoadTarget;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.DigraphQueryEvalResult;
import com.google.devtools.build.lib.query2.engine.MinDepthUniquifier;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryExpressionContext;
import com.google.devtools.build.lib.query2.engine.QueryUtil.MinDepthUniquifierImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.MutableKeyExtractorBackedMapImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.ThreadSafeMutableKeyExtractorBackedSetImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.UniquifierImpl;
import com.google.devtools.build.lib.query2.engine.SkyframeRestartQueryException;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.lib.skyframe.SkyframeLabelVisitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The environment of a Blaze query. Not thread-safe.
 */
public class BlazeQueryEnvironment extends AbstractBlazeQueryEnvironment<Target> {
  private static final int MAX_DEPTH_FULL_SCAN_LIMIT = 20;
  private final Map<String, Collection<Target>> resolvedTargetPatterns = new HashMap<>();
  private final TargetPatternPreloader targetPatternPreloader;
  private final PathFragment relativeWorkingDirectory;
  private final TransitivePackageLoader transitivePackageLoader;
  private final TargetProvider targetProvider;
  private final CachingPackageLocator cachingPackageLocator;
  private final Digraph<Target> graph = new Digraph<>();
  private final ErrorPrintingTargetEdgeErrorObserver errorObserver;
  private final LabelVisitor labelVisitor;
  protected final int loadingPhaseThreads;

  private final BlazeTargetAccessor accessor = new BlazeTargetAccessor(this);

  /**
   * Note that the correct operation of this class critically depends on the Reporter being a
   * singleton object, shared by all cooperating classes contributing to Query.
   *
   * @param strictScope if true, fail the whole query if a label goes out of scope.
   * @param loadingPhaseThreads the number of threads to use during loading the packages for the
   *     query.
   * @param labelFilter a predicate that determines if a specific label is allowed to be visited
   *     during query execution. If it returns false, the query execution is stopped with an error
   *     message.
   * @param settings a set of enabled settings
   */
  public BlazeQueryEnvironment(
      TransitivePackageLoader transitivePackageLoader,
      TargetProvider targetProvider,
      CachingPackageLocator cachingPackageLocator,
      TargetPatternPreloader targetPatternPreloader,
      PathFragment relativeWorkingDirectory,
      boolean keepGoing,
      boolean strictScope,
      int loadingPhaseThreads,
      Predicate<Label> labelFilter,
      ExtendedEventHandler eventHandler,
      Set<Setting> settings,
      Iterable<QueryFunction> extraFunctions) {
    super(keepGoing, strictScope, labelFilter, eventHandler, settings, extraFunctions);
    this.targetPatternPreloader = targetPatternPreloader;
    this.relativeWorkingDirectory = relativeWorkingDirectory;
    this.transitivePackageLoader = transitivePackageLoader;
    this.targetProvider = targetProvider;
    this.cachingPackageLocator = cachingPackageLocator;
    this.errorObserver = new ErrorPrintingTargetEdgeErrorObserver(this.eventHandler);
    this.loadingPhaseThreads = loadingPhaseThreads;
    this.labelVisitor = new LabelVisitor(targetProvider, dependencyFilter);
  }

  @Override
  public void close() {
    // BlazeQueryEnvironment has no resources that need to be cleaned up.
  }

  @Override
  public DigraphQueryEvalResult<Target> evaluateQuery(
      QueryExpression expr,
      ThreadSafeOutputFormatterCallback<Target> callback)
          throws QueryException, InterruptedException, IOException {
    resolvedTargetPatterns.clear();
    QueryEvalResult queryEvalResult = super.evaluateQuery(expr, callback);
    return new DigraphQueryEvalResult<>(
        queryEvalResult.getSuccess(), queryEvalResult.isEmpty(), graph);
  }

  @Override
  public Collection<Target> getSiblingTargetsInPackage(Target target) {
    Collection<Target> siblings = target.getPackage().getTargets().values();
    // Ensure that the sibling targets are in the graph being built-up.
    siblings.forEach(this::getNode);
    return siblings;
  }

  @Override
  public QueryTaskFuture<Void> getTargetsMatchingPattern(
      QueryExpression owner, String pattern, Callback<Target> callback) {
    try {
      getTargetsMatchingPatternImpl(pattern, callback);
      return immediateSuccessfulFuture(null);
    } catch (QueryException e) {
      return immediateFailedFuture(e);
    } catch (InterruptedException e) {
      return immediateCancelledFuture();
    }
  }

  private void getTargetsMatchingPatternImpl(String pattern, Callback<Target> callback)
      throws QueryException, InterruptedException {
    // We can safely ignore the boolean error flag. The evaluateQuery() method above wraps the
    // entire query computation in an error sensor.

    // This must be a collections class with a fast contains() implementation, or the code below
    // becomes quadratic in runtime.
    Set<Target> targets = new LinkedHashSet<>(resolvedTargetPatterns.get(pattern));

    // Sets.filter would be more convenient here, but can't deal with exceptions.
    if (labelFilter != Predicates.<Label>alwaysTrue()) {
      // The labelFilter is always true for bazel query; it's only used for genquery rules.
      Iterator<Target> targetIterator = targets.iterator();
      while (targetIterator.hasNext()) {
        Target target = targetIterator.next();
        if (!validateScope(target.getLabel(), strictScope)) {
          targetIterator.remove();
        }
      }
    }

    Set<PathFragment> packages = CompactHashSet.create();
    for (Target target : targets) {
      packages.add(target.getLabel().getPackageFragment());
    }

    for (Target target : targets) {
      // This triggers node creation in the Digraph; getOrCreate(X) returns X.
      getOrCreate(target);

      // Preservation of graph order: it is important that targets obtained via
      // a wildcard such as p:* are correctly ordered w.r.t. each other, so to
      // ensure this, we add edges between any pair of directly connected
      // targets in this set.
      if (target instanceof OutputFile) {
        OutputFile outputFile = (OutputFile) target;
        if (targets.contains(outputFile.getGeneratingRule())) {
          makeEdge(outputFile, outputFile.getGeneratingRule());
        }
      } else if (target instanceof Rule) {
        Rule rule = (Rule) target;
        for (Label label : rule.getLabels(dependencyFilter)) {
          if (!packages.contains(label.getPackageFragment())) {
            continue;  // don't cause additional package loading
          }
          try {
            if (!validateScope(label, strictScope)) {
              continue;  // Don't create edges to targets which are out of scope.
            }
            Target to = getTargetOrThrow(label);
            if (targets.contains(to)) {
              makeEdge(rule, to);
            }
          } catch (NoSuchThingException e) {
            /* ignore */
          }
        }
      }
    }
    callback.process(targets);
  }

  @Override
  public Target getTarget(Label label)
      throws TargetNotFoundException, QueryException, InterruptedException {
    // Can't use strictScope here because we are expecting a target back.
    validateScope(label, true);
    try {
      return getNode(getTargetOrThrow(label)).getLabel();
    } catch (NoSuchThingException e) {
      throw new TargetNotFoundException(e);
    }
  }

  private Node<Target> getNode(Target target) {
    return graph.createNode(target);
  }

  private Collection<Node<Target>> getNodes(Iterable<Target> target) {
    Set<Node<Target>> result = new LinkedHashSet<>();
    for (Target t : target) {
      result.add(getNode(t));
    }
    return result;
  }

  @Override
  public Target getOrCreate(Target target) {
    return getNode(target).getLabel();
  }

  @Override
  public Collection<Target> getFwdDeps(
      Iterable<Target> targets, QueryExpressionContext<Target> context) {
    ThreadSafeMutableSet<Target> result = createThreadSafeMutableSet();
    for (Target target : targets) {
      result.addAll(getTargetsFromNodes(getNode(target).getSuccessors()));
    }
    return result;
  }

  @Override
  public Collection<Target> getReverseDeps(
      Iterable<Target> targets, QueryExpressionContext<Target> context) {
    ThreadSafeMutableSet<Target> result = createThreadSafeMutableSet();
    for (Target target : targets) {
      result.addAll(getTargetsFromNodes(getNode(target).getPredecessors()));
    }
    return result;
  }

  @Override
  public ThreadSafeMutableSet<Target> getTransitiveClosure(
      ThreadSafeMutableSet<Target> targetNodes, QueryExpressionContext<Target> context) {
    for (Target node : targetNodes) {
      checkBuilt(node);
    }
    return getTargetsFromNodes(graph.getFwdReachable(getNodes(targetNodes)));
  }

  /**
   * Checks that the graph rooted at 'targetNode' has been completely built;
   * fails if not.  Callers of {@link #getTransitiveClosure} must ensure that
   * {@link #buildTransitiveClosure} has been called before.
   *
   * <p>It would be inefficient and failure-prone to make getTransitiveClosure
   * call buildTransitiveClosure directly.  Also, it would cause
   * nondeterministic behavior of the operators, since the set of packages
   * loaded (and hence errors reported) would depend on the ordering details of
   * the query operators' implementations.
   */
  private void checkBuilt(Target targetNode) {
    Preconditions.checkState(
        labelVisitor.hasVisited(targetNode.getLabel()),
        "getTransitiveClosure(%s) called without prior call to buildTransitiveClosure()",
        targetNode);
  }

  @Override
  public void buildTransitiveClosure(QueryExpression caller,
                                     ThreadSafeMutableSet<Target> targetNodes,
                                     int maxDepth) throws QueryException, InterruptedException {
    try (SilentCloseable closeable = Profiler.instance().profile("preloadTransitiveClosure")) {
      preloadTransitiveClosure(targetNodes, maxDepth);
    }
    try (SilentCloseable closeable = Profiler.instance().profile("syncWithVisitor")) {
      labelVisitor.syncWithVisitor(
          eventHandler,
          targetNodes,
          keepGoing,
          loadingPhaseThreads,
          maxDepth,
          errorObserver,
          new GraphBuildingObserver());
    }

    if (errorObserver.hasErrors()) {
      reportBuildFileError(caller, "errors were encountered while computing transitive closure");
    }
  }

  @Override
  public Iterable<Target> getNodesOnPath(
      Target from, Target to, QueryExpressionContext<Target> context) {
    ImmutableList.Builder<Target> builder = ImmutableList.builder();
    for (Node<Target> node : graph.getShortestPath(getNode(from), getNode(to))) {
      builder.add(node.getLabel());
    }
    return builder.build();
  }

  @ThreadSafe
  @Override
  public ThreadSafeMutableSet<Target> createThreadSafeMutableSet() {
    return new ThreadSafeMutableKeyExtractorBackedSetImpl<>(
        TargetKeyExtractor.INSTANCE, Target.class);
  }

  @Override
  public <V> MutableMap<Target, V> createMutableMap() {
    return new MutableKeyExtractorBackedMapImpl<>(TargetKeyExtractor.INSTANCE);
  }

  @Override
  public Uniquifier<Target> createUniquifier() {
    return new UniquifierImpl<>(TargetKeyExtractor.INSTANCE);
  }

  @Override
  public MinDepthUniquifier<Target> createMinDepthUniquifier() {
    return new MinDepthUniquifierImpl<>(TargetKeyExtractor.INSTANCE, /*concurrencyLevel=*/ 1);
  }

  private void preloadTransitiveClosure(ThreadSafeMutableSet<Target> targets, int maxDepth)
      throws InterruptedException {
    if (maxDepth >= MAX_DEPTH_FULL_SCAN_LIMIT && transitivePackageLoader != null) {
      // Only do the full visitation if "maxDepth" is large enough. Otherwise, the benefits of
      // preloading will be outweighed by the cost of doing more work than necessary.
      Set<Label> labels = targets.stream().map(Target::getLabel).collect(toImmutableSet());
      ((SkyframeLabelVisitor) transitivePackageLoader)
          .sync(eventHandler, labels, keepGoing, loadingPhaseThreads, /* errorOnCycles= */ false);
    }
  }

  /**
   * It suffices to synchronize the modifications of this.graph from within the
   * GraphBuildingObserver, because that's the only concurrent part.
   * Concurrency is always encapsulated within the evaluation of a single query
   * operator (e.g. deps(), somepath(), etc).
   */
  private class GraphBuildingObserver implements TargetEdgeObserver {

    @Override
    public void edge(Target from, Attribute attribute, Target to) {
      Preconditions.checkState(attribute == null ||
          dependencyFilter.apply(((Rule) from), attribute),
          "Disallowed edge from LabelVisitor: %s --> %s", from, to);
      makeEdge(from, to);
    }

    @Override
    public void node(Target node) {
      graph.createNode(node);
    }

    @Override
    public void missingEdge(Target target, Label to, NoSuchThingException e) {
      // No - op.
    }
  }

  private void makeEdge(Target from, Target to) {
    graph.addEdge(from, to);
  }

  private Target getTargetOrThrow(Label label)
      throws NoSuchThingException, SkyframeRestartQueryException, InterruptedException {
    Target target = targetProvider.getTarget(eventHandler, label);
    if (target == null) {
      throw new SkyframeRestartQueryException();
    }
    return target;
  }

  // TODO(bazel-team): rename this to getDependentFiles when all implementations
  // of QueryEnvironment is fixed.
  @Override
  public ThreadSafeMutableSet<Target> getBuildFiles(
      final QueryExpression caller,
      ThreadSafeMutableSet<Target> nodes,
      boolean buildFiles,
      boolean loads,
      QueryExpressionContext<Target> context)
      throws QueryException {
    ThreadSafeMutableSet<Target> dependentFiles = createThreadSafeMutableSet();
    Set<PackageIdentifier> seenPackages = new HashSet<>();
    // Keep track of seen labels, to avoid adding a fake subinclude label that also exists as a
    // real target.
    Set<Label> seenLabels = new HashSet<>();

    // Adds all the package definition files (BUILD files and build
    // extensions) for package "pkg", to "buildfiles".
    for (Target x : nodes) {
      Package pkg = x.getPackage();
      if (seenPackages.add(pkg.getPackageIdentifier())) {
        if (buildFiles) {
          addIfUniqueLabel(getNode(pkg.getBuildFile()), seenLabels, dependentFiles);
        }

        List<Label> extensions = new ArrayList<>();
        if (loads) {
          extensions.addAll(pkg.getStarlarkFileDependencies());
        }

        for (Label extension : extensions) {

          Node<Target> loadTarget = getLoadTarget(extension, pkg);
          addIfUniqueLabel(loadTarget, seenLabels, dependentFiles);

          // Also add the BUILD file of the extension.
          if (buildFiles) {
            Path buildFileForLoad =
                cachingPackageLocator.getBuildFileForPackage(
                    loadTarget.getLabel().getLabel().getPackageIdentifier());
            if (buildFileForLoad != null) {
              Label buildFileLabel =
                  Label.createUnvalidated(
                      loadTarget.getLabel().getLabel().getPackageIdentifier(),
                      buildFileForLoad.getBaseName());
              addIfUniqueLabel(
                  getNode(new FakeLoadTarget(buildFileLabel, pkg)), seenLabels, dependentFiles);
            }
          }
        }
      }
    }
    return dependentFiles;
  }

  @Override
  protected void preloadOrThrow(QueryExpression caller, Collection<String> patterns)
      throws TargetParsingException, InterruptedException {
    if (!resolvedTargetPatterns.keySet().containsAll(patterns)) {
      // Note that this may throw a RuntimeException if deps are missing in Skyframe and this is
      // being called from within a SkyFunction.
      resolvedTargetPatterns.putAll(
          targetPatternPreloader.preloadTargetPatterns(
              eventHandler, relativeWorkingDirectory, patterns, keepGoing));
    }
  }

  private static void addIfUniqueLabel(Node<Target> node, Set<Label> labels, Set<Target> nodes) {
    if (labels.add(node.getLabel().getLabel())) {
      nodes.add(node.getLabel());
    }
  }

  private Node<Target> getLoadTarget(Label label, Package pkg) {
    return getNode(new FakeLoadTarget(label, pkg));
  }

  @Override
  public TargetAccessor<Target> getAccessor() {
    return accessor;
  }

  /** Given a set of target nodes, returns the targets. */
  private ThreadSafeMutableSet<Target> getTargetsFromNodes(Iterable<Node<Target>> input) {
    ThreadSafeMutableSet<Target> result = createThreadSafeMutableSet();
    for (Node<Target> node : input) {
      result.add(node.getLabel());
    }
    return result;
  }
}
