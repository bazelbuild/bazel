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

import com.google.common.base.Preconditions;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern.Parser;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.TargetEdgeObserver;
import com.google.devtools.build.lib.pkgcache.TargetPatternPreloader;
import com.google.devtools.build.lib.pkgcache.TargetProvider;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.common.QueryTransitivePackagePreloader;
import com.google.devtools.build.lib.query2.compat.FakeLoadTarget;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.MinDepthUniquifier;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.CustomFunctionQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryExpressionContext;
import com.google.devtools.build.lib.query2.engine.QueryUtil.MinDepthUniquifierImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.ThreadSafeMutableKeyExtractorBackedSetImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.UniquifierImpl;
import com.google.devtools.build.lib.query2.engine.SamePkgDirectRdepsFunction;
import com.google.devtools.build.lib.query2.engine.SkyframeRestartQueryException;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.OptionalInt;
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
 *
 * <p>This environment is valid only for a single query, called via {@link #evaluateQuery}. Call
 * only once!
 */
public class GraphlessBlazeQueryEnvironment extends AbstractBlazeQueryEnvironment<Target>
    implements CustomFunctionQueryEnvironment<Target> {
  private static final int MAX_DEPTH_FULL_SCAN_LIMIT = 20;
  private final Map<String, Collection<Target>> resolvedTargetPatterns = new HashMap<>();
  private final TargetPatternPreloader targetPatternPreloader;
  private final TargetPattern.Parser mainRepoTargetParser;
  @Nullable private final QueryTransitivePackagePreloader queryTransitivePackagePreloader;
  private final TargetProvider targetProvider;
  private final CachingPackageLocator cachingPackageLocator;
  private final ErrorPrintingTargetEdgeErrorObserver errorObserver;
  private final LabelVisitor labelVisitor;
  protected final int loadingPhaseThreads;

  private final BlazeTargetAccessor accessor = new BlazeTargetAccessor(this);

  private boolean doneQuery = false;

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
      @Nullable QueryTransitivePackagePreloader queryTransitivePackagePreloader,
      TargetProvider targetProvider,
      CachingPackageLocator cachingPackageLocator,
      TargetPatternPreloader targetPatternPreloader,
      Parser mainRepoTargetParser,
      boolean keepGoing,
      boolean strictScope,
      int loadingPhaseThreads,
      Predicate<Label> labelFilter,
      ExtendedEventHandler eventHandler,
      Set<Setting> settings,
      Iterable<QueryFunction> extraFunctions,
      LabelPrinter labelPrinter) {
    super(
        keepGoing, strictScope, labelFilter, eventHandler, settings, extraFunctions, labelPrinter);
    this.targetPatternPreloader = targetPatternPreloader;
    this.mainRepoTargetParser = mainRepoTargetParser;
    this.queryTransitivePackagePreloader = queryTransitivePackagePreloader;
    this.targetProvider = targetProvider;
    this.cachingPackageLocator = cachingPackageLocator;
    this.errorObserver = new ErrorPrintingTargetEdgeErrorObserver(this.eventHandler);
    this.loadingPhaseThreads = loadingPhaseThreads;
    this.labelVisitor = new LabelVisitor(targetProvider, dependencyFilter);
  }

  @Override
  public QueryTaskFuture<Void> eval(
      QueryExpression expr,
      QueryExpressionContext<Target> context,
      Callback<Target> callback,
      boolean batch) {
    if (batch) {
      // This uses AbstractBlazeQueryEnvironment#eval that aggregates the results of the futures
      // into a single batch before running the callback on the batch of results, providing an
      // alternative for the environment to decide when to batch the results and when batching is
      // not needed.
      return super.eval(expr, context, callback);
    }
    return eval(expr, context, callback);
  }

  @Override
  public QueryTaskFuture<Void> eval(
      QueryExpression expr, QueryExpressionContext<Target> context, Callback<Target> callback) {
    // The graphless query implementation does not perform any streaming at this point. However,
    // not all operators return a single future (e.g. 'SetExpression'), as such, do not use this if
    // the callback does heavy blocking work (e.g. 'deps').
    return expr.eval(this, context, callback);
  }

  @Override
  public QueryEvalResult evaluateQuery(
      QueryExpression expr, ThreadSafeOutputFormatterCallback<Target> callback)
      throws QueryException, IOException, InterruptedException {
    Preconditions.checkState(!doneQuery, "Can only use environment for one query: %s", expr);
    doneQuery = true;
    return evaluateQueryInternal(expr, callback);
  }

  @Override
  public void close() {
    // BlazeQueryEnvironment has no resources that need to be cleaned up.
  }

  @Override
  public Collection<Target> getSiblingTargetsInPackage(Target target) {
    // TODO(https://github.com/bazelbuild/bazel/issues/23852): support lazy macro expansion
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
    Set<Target> targets = new LinkedHashSet<>(resolvedTargetPatterns.get(pattern));
    validateScopeOfTargets(targets);
    callback.process(targets);
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
      Iterable<Target> from,
      OptionalInt maxDepth,
      QueryExpression caller,
      Callback<Target> callback)
      throws InterruptedException, QueryException {
    // TODO(ulfjack): There's no need to visit the transitive closure twice. Ideally, preloading
    //  would return the list of targets, but it currently only returns the list of labels.
    try (SilentCloseable closeable = Profiler.instance().profile("preloadTransitiveClosure")) {
      preloadTransitiveClosure(from, maxDepth, caller);
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
    try (SilentCloseable closeable = Profiler.instance().profile("preloadTransitiveClosure")) {
      preloadTransitiveClosure(from, /*maxDepth=*/ OptionalInt.empty(), caller);
    }
    Iterable<Target> results =
        new PathLabelVisitor(targetProvider, dependencyFilter, errorObserver)
            .somePath(eventHandler, from, to);
    if (errorObserver.hasErrors()) {
      handleError(
          caller,
          "errors were encountered while computing transitive closure",
          errorObserver.getDetailedExitCode());
    }
    callback.process(results);
  }

  @Override
  public void allPaths(
      Iterable<Target> from, Iterable<Target> to, QueryExpression caller, Callback<Target> callback)
      throws InterruptedException, QueryException {
    try (SilentCloseable closeable = Profiler.instance().profile("preloadTransitiveClosure")) {
      preloadTransitiveClosure(from, /*maxDepth=*/ OptionalInt.empty(), caller);
    }
    Iterable<Target> results =
        new PathLabelVisitor(targetProvider, dependencyFilter, errorObserver)
            .allPaths(eventHandler, from, to);
    if (errorObserver.hasErrors()) {
      handleError(
          caller,
          "errors were encountered while computing transitive closure",
          errorObserver.getDetailedExitCode());
    }
    callback.process(results);
  }

  @Override
  public void samePkgDirectRdeps(
      Iterable<Target> from, QueryExpression caller, Callback<Target> callback)
      throws InterruptedException, QueryException {
    Set<Target> targetsToPreload = new HashSet<>();
    for (Target t : from) {
      targetsToPreload.addAll(getSiblingTargetsInPackage(t));
    }
    try (SilentCloseable closeable = Profiler.instance().profile("preloadTransitiveClosure")) {
      preloadTransitiveClosure(
          targetsToPreload, /*maxDepth=*/ SamePkgDirectRdepsFunction.DEPTH_ONE, caller);
    }
    Iterable<Target> results =
        new PathLabelVisitor(targetProvider, dependencyFilter, errorObserver)
            .samePkgDirectRdeps(eventHandler, from);
    if (errorObserver.hasErrors()) {
      handleError(
          caller,
          "errors were encountered while computing transitive closure",
          errorObserver.getDetailedExitCode());
    }
    callback.process(results);
  }

  @Override
  public void rdeps(
      Iterable<Target> from,
      Iterable<Target> universe,
      OptionalInt maxDepth,
      QueryExpression caller,
      Callback<Target> callback)
      throws InterruptedException, QueryException {
    try (SilentCloseable closeable = Profiler.instance().profile("preloadTransitiveClosure")) {
      preloadTransitiveClosure(
          universe,
          // PathLabelVisitor#rdeps, called below, necessarily needs to crawl the full DTC of
          // `universe` in order to be able to reify the reverse edges needed to determine the rdeps
          // of `from` at the specified depth. Therefore we preload the full DTC of `universe` in
          // parallel, so that PathLabelVisitor#rdeps doesn't need to do novel package loading.
          /* maxDepth= */ OptionalInt.empty(),
          caller);
    }
    Iterable<Target> results =
        new PathLabelVisitor(targetProvider, dependencyFilter, errorObserver)
            .rdeps(eventHandler, from, universe, maxDepth);
    if (errorObserver.hasErrors()) {
      handleError(
          caller,
          "errors were encountered while computing transitive closure",
          errorObserver.getDetailedExitCode());
    }
    callback.process(results);
  }

  @Override
  public void buildTransitiveClosure(
      QueryExpression caller, ThreadSafeMutableSet<Target> targetNodes, OptionalInt maxDepth)
      throws QueryException, InterruptedException {
    try (SilentCloseable closeable = Profiler.instance().profile("preloadTransitiveClosure")) {
      preloadTransitiveClosure(targetNodes, maxDepth, caller);
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
  public Uniquifier<Target> createUniquifier() {
    return new UniquifierImpl<>(TargetKeyExtractor.INSTANCE);
  }

  @Override
  public MinDepthUniquifier<Target> createMinDepthUniquifier() {
    return new MinDepthUniquifierImpl<>(TargetKeyExtractor.INSTANCE, /*concurrencyLevel=*/ 1);
  }

  @Override
  public TransitiveLoadFilesHelper<Target> getTransitiveLoadFilesHelper() {
    return new TransitiveLoadFilesHelperForTargets() {
      @Override
      public Target getLoadFileTarget(Target originalTarget, Label bzlLabel) {
        // TODO(https://github.com/bazelbuild/bazel/issues/23852): support lazy macro expansion
        return new FakeLoadTarget(bzlLabel, originalTarget.getPackage());
      }

      @Nullable
      @Override
      public Target maybeGetBuildFileTargetForLoadFileTarget(
          Target originalTarget, Label bzlLabel) {
        PackageIdentifier pkgIdOfBzlLabel = bzlLabel.getPackageIdentifier();
        String baseName = cachingPackageLocator.getBaseNameForLoadedPackage(pkgIdOfBzlLabel);
        if (baseName == null) {
          return null;
        }
        // TODO(https://github.com/bazelbuild/bazel/issues/23852): support lazy macro expansion
        return new FakeLoadTarget(
            Label.createUnvalidated(pkgIdOfBzlLabel, baseName), originalTarget.getPackage());
      }
    };
  }

  private void preloadTransitiveClosure(
      Iterable<Target> targets, OptionalInt maxDepth, QueryExpression callerForError)
      throws InterruptedException, QueryException {
    if (QueryEnvironment.shouldVisit(maxDepth, MAX_DEPTH_FULL_SCAN_LIMIT)
        && queryTransitivePackagePreloader != null) {
      // Only do the full visitation if "maxDepth" is large enough. Otherwise, the benefits of
      // preloading will be outweighed by the cost of doing more work than necessary.
      Set<Label> labels = CompactHashSet.create();
      for (Target t : targets) {
        labels.add(t.getLabel());
      }
      queryTransitivePackagePreloader.preloadTransitiveTargets(
          eventHandler,
          labels,
          keepGoing,
          loadingPhaseThreads,
          // Don't throw an error if in keep-going mode or if the depth was limited: it's possible
          // that an encountered error was deeper than the depth bound.
          keepGoing || maxDepth.isPresent() ? null : callerForError);
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
  protected void preloadOrThrow(QueryExpression caller, Collection<String> patterns)
      throws TargetParsingException, InterruptedException {
    Preconditions.checkState(
        resolvedTargetPatterns.isEmpty(),
        "Already resolved patterns: %s %s",
        patterns,
        resolvedTargetPatterns);
    // Note that this may throw a RuntimeException if deps are missing in Skyframe and this is
    // being called from within a SkyFunction.
    resolvedTargetPatterns.putAll(
        targetPatternPreloader.preloadTargetPatterns(
            eventHandler, mainRepoTargetParser, patterns, keepGoing));
  }

  @Override
  public TargetAccessor<Target> getAccessor() {
    return accessor;
  }
}
