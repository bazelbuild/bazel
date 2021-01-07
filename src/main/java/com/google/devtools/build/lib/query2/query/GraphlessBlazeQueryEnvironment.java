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
package com.google.devtools.build.lib.query2.query;

import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.QueryTransitivePackagePreloader;
import com.google.devtools.build.lib.pkgcache.TargetEdgeObserver;
import com.google.devtools.build.lib.pkgcache.TargetPatternPreloader;
import com.google.devtools.build.lib.pkgcache.TargetProvider;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.compat.FakeLoadTarget;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.MinDepthUniquifier;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.CustomFunctionQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryExpressionContext;
import com.google.devtools.build.lib.query2.engine.QueryUtil.MinDepthUniquifierImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.MutableKeyExtractorBackedMapImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.ThreadSafeMutableKeyExtractorBackedSetImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.UniquifierImpl;
import com.google.devtools.build.lib.query2.engine.SkyframeRestartQueryException;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * The environment of a Blaze query. Not thread-safe.
 *
 * <p>In contrast with {@link BlazeQueryEnvironment}, this one does not support ordered output, and
 * therefore also does not make a partial copy of the graph in a Digraph instance. As a corollary,
 * it only returns an instance of {@link
 * com.google.devtools.build.lib.query2.engine.QueryEvalResult} rather than {@link
 * com.google.devtools.build.lib.query2.engine.DigraphQueryEvalResult}, and can therefore not be
 * used with most existing {@link com.google.devtools.build.lib.query2.query.output.OutputFormatter}
 * implementations, many of which expect the latter.
 */
public class GraphlessBlazeQueryEnvironment extends AbstractBlazeQueryEnvironment<Target>
    implements CustomFunctionQueryEnvironment<Target> {
  private static final int MAX_DEPTH_FULL_SCAN_LIMIT = 20;
  private final Map<String, Collection<Target>> resolvedTargetPatterns = new HashMap<>();
  private final TargetPatternPreloader targetPatternPreloader;
  private final PathFragment relativeWorkingDirectory;
  private final QueryTransitivePackagePreloader queryTransitivePackagePreloader;
  private final TargetProvider targetProvider;
  private final CachingPackageLocator cachingPackageLocator;
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
  public GraphlessBlazeQueryEnvironment(
      QueryTransitivePackagePreloader queryTransitivePackagePreloader,
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
    this.queryTransitivePackagePreloader = queryTransitivePackagePreloader;
    this.targetProvider = targetProvider;
    this.cachingPackageLocator = cachingPackageLocator;
    this.errorObserver = new ErrorPrintingTargetEdgeErrorObserver(this.eventHandler);
    this.loadingPhaseThreads = loadingPhaseThreads;
    this.labelVisitor = new LabelVisitor(targetProvider, dependencyFilter);
  }

  @Override
  public QueryTaskFuture<Void> eval(
      QueryExpression expr, QueryExpressionContext<Target> context, Callback<Target> callback) {
    // The graphless query implementation does not perform any streaming at this point, and all
    // operators only make a single call to the callback, so it is perfectly safe to pass the
    // callback to the expression eval call. This is also a lot cheaper than making a copy here.
    return expr.eval(this, context, callback);
  }

  @Override
  public void close() {
    // BlazeQueryEnvironment has no resources that need to be cleaned up.
  }

  @Override
  public Collection<Target> getSiblingTargetsInPackage(Target target) {
    return target.getPackage().getTargets().values();
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
    callback.process(resolvedTargetPatterns.get(pattern));
  }

  @Override
  public Target getTarget(Label label)
      throws TargetNotFoundException, QueryException, InterruptedException {
    try {
      return getTargetOrThrow(label);
    } catch (NoSuchThingException e) {
      throw new TargetNotFoundException(e, e.getDetailedExitCode());
    }
  }

  @Override
  public Target getOrCreate(Target target) {
    return target;
  }

  @Override
  public Collection<Target> getFwdDeps(
      Iterable<Target> targets, QueryExpressionContext<Target> context) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Collection<Target> getReverseDeps(
      Iterable<Target> targets, QueryExpressionContext<Target> context) {
    throw new UnsupportedOperationException();
  }

  @Override
  public ThreadSafeMutableSet<Target> getTransitiveClosure(
      ThreadSafeMutableSet<Target> targetNodes, QueryExpressionContext<Target> context) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void deps(
      Iterable<Target> from, int maxDepth, QueryExpression caller, Callback<Target> callback)
      throws InterruptedException, QueryException {
    // TODO(ulfjack): There's no need to visit the transitive closure twice. Ideally, preloading
    //  would return the list of targets, but it currently only returns the list of labels.
    try (SilentCloseable closeable = Profiler.instance().profile("preloadTransitiveClosure")) {
      preloadTransitiveClosure(from, maxDepth);
    }
    Set<Target> result = Sets.newConcurrentHashSet();
    try (SilentCloseable closeable = Profiler.instance().profile("syncUncached")) {
      new LabelVisitor(targetProvider, dependencyFilter)
          .syncUncached(
              eventHandler,
              from,
              keepGoing,
              loadingPhaseThreads,
              maxDepth,
              new TargetEdgeObserver() {
                @Override
                public void edge(Target from, Attribute attribute, Target to) {
                  errorObserver.edge(from, attribute, to);
                }

                @Override
                public void missingEdge(@Nullable Target target, Label to, NoSuchThingException e) {
                  errorObserver.missingEdge(target, to, e);
                }

                @Override
                public void node(Target node) {
                  result.add(node);
                  errorObserver.node(node);
                }
              });
    }
    if (errorObserver.hasErrors()) {
      handleError(
          caller,
          "errors were encountered while computing transitive closure",
          errorObserver.getDetailedExitCode());
    }
    callback.process(result);
  }

  @Override
  public void somePath(
      Iterable<Target> from, Iterable<Target> to, QueryExpression caller, Callback<Target> callback)
      throws InterruptedException, QueryException {
    try {
      callback.process(
          new PathLabelVisitor(targetProvider, dependencyFilter).somePath(eventHandler, from, to));
    } catch (NoSuchThingException e) {
      throw new QueryException(
          caller, e.getMessage(), e, e.getDetailedExitCode().getFailureDetail());
    }
  }

  @Override
  public void allPaths(
      Iterable<Target> from, Iterable<Target> to, QueryExpression caller, Callback<Target> callback)
      throws InterruptedException, QueryException {
    try {
      callback.process(
          new PathLabelVisitor(targetProvider, dependencyFilter).allPaths(eventHandler, from, to));
    } catch (NoSuchThingException e) {
      throw new QueryException(caller, e.getMessage(), e.getDetailedExitCode().getFailureDetail());
    }
  }

  @Override
  public void samePkgDirectRdeps(
      Iterable<Target> from, QueryExpression caller, Callback<Target> callback)
      throws InterruptedException, QueryException {
    try {
      callback.process(
          new PathLabelVisitor(targetProvider, dependencyFilter)
              .samePkgDirectRdeps(eventHandler, from));
    } catch (NoSuchThingException e) {
      throw new QueryException(caller, e.getMessage(), e.getDetailedExitCode().getFailureDetail());
    }
  }

  @Override
  public void rdeps(
      Iterable<Target> from,
      Iterable<Target> universe,
      int maxDepth,
      QueryExpression caller,
      Callback<Target> callback)
      throws InterruptedException, QueryException {
    try {
      callback.process(
          new PathLabelVisitor(targetProvider, dependencyFilter)
              .rdeps(eventHandler, from, universe, maxDepth));
    } catch (NoSuchThingException e) {
      throw new QueryException(caller, e.getMessage(), e.getDetailedExitCode().getFailureDetail());
    }
  }

  @Override
  public void buildTransitiveClosure(
      QueryExpression caller, ThreadSafeMutableSet<Target> targetNodes, int maxDepth)
      throws QueryException, InterruptedException {
    try (SilentCloseable closeable = Profiler.instance().profile("preloadTransitiveClosure")) {
      preloadTransitiveClosure(targetNodes, maxDepth);
    }
    try (SilentCloseable closeable = Profiler.instance().profile("syncWithVisitor")) {
      labelVisitor.syncWithVisitor(
          eventHandler, targetNodes, keepGoing, loadingPhaseThreads, maxDepth, errorObserver);
    }

    if (errorObserver.hasErrors()) {
      handleError(
          caller,
          "errors were encountered while computing transitive closure",
          errorObserver.getDetailedExitCode());
    }
  }

  @Override
  public Iterable<Target> getNodesOnPath(
      Target from, Target to, QueryExpressionContext<Target> context) {
    throw new UnsupportedOperationException();
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

  private void preloadTransitiveClosure(Iterable<Target> targets, int maxDepth)
      throws InterruptedException {
    if (maxDepth >= MAX_DEPTH_FULL_SCAN_LIMIT && queryTransitivePackagePreloader != null) {
      // Only do the full visitation if "maxDepth" is large enough. Otherwise, the benefits of
      // preloading will be outweighed by the cost of doing more work than necessary.
      Set<Label> labels = CompactHashSet.create();
      for (Target t : targets) {
        labels.add(t.getLabel());
      }
      queryTransitivePackagePreloader.preloadTransitiveTargets(
          eventHandler, labels, keepGoing, loadingPhaseThreads);
    }
  }

  private Target getTargetOrThrow(Label label)
      throws NoSuchThingException, SkyframeRestartQueryException, InterruptedException {
    Target target = targetProvider.getTarget(eventHandler, label);
    if (target == null) {
      throw new SkyframeRestartQueryException();
    }
    return target;
  }

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

    // Adds all the package definition files (BUILD files and build extensions) for package "pkg",
    // to dependentFiles.
    for (Target x : nodes) {
      Package pkg = x.getPackage();
      if (seenPackages.add(pkg.getPackageIdentifier())) {
        if (buildFiles) {
          dependentFiles.add(pkg.getBuildFile());
        }

        List<Label> extensions = new ArrayList<>();
        if (loads) {
          extensions.addAll(pkg.getStarlarkFileDependencies());
        }

        for (Label extension : extensions) {
          Target loadTarget = new FakeLoadTarget(extension, pkg);
          dependentFiles.add(loadTarget);

          // Also add the BUILD file of the extension.
          if (buildFiles) {
            Path buildFileForLoad =
                cachingPackageLocator.getBuildFileForPackage(
                    loadTarget.getLabel().getPackageIdentifier());
            if (buildFileForLoad != null) {
              Label buildFileLabel =
                  Label.createUnvalidated(
                      loadTarget.getLabel().getPackageIdentifier(), buildFileForLoad.getBaseName());
              dependentFiles.add(new FakeLoadTarget(buildFileLabel, pkg));
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

  @Override
  public TargetAccessor<Target> getAccessor() {
    return accessor;
  }
}
