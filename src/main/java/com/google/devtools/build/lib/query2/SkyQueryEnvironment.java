// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.pkgcache.FilteringPolicies.NO_FILTER;

import com.google.common.base.Ascii;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Throwables;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.AsyncFunction;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.concurrent.BlockingStack;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.concurrent.ParallelVisitor.VisitTaskStatusCallback;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.DelegatingEventHandler;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.compat.FakeLoadTarget;
import com.google.devtools.build.lib.query2.engine.AllRdepsFunction;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.KeyExtractor;
import com.google.devtools.build.lib.query2.engine.MinDepthUniquifier;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryExpressionContext;
import com.google.devtools.build.lib.query2.engine.QueryExpressionMapper;
import com.google.devtools.build.lib.query2.engine.QueryUtil.MinDepthUniquifierImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.MutableKeyExtractorBackedMapImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.NonExceptionalUniquifier;
import com.google.devtools.build.lib.query2.engine.QueryUtil.ThreadSafeMutableKeyExtractorBackedSetImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.UniquifierImpl;
import com.google.devtools.build.lib.query2.engine.StreamableQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.lib.query2.query.BlazeTargetAccessor;
import com.google.devtools.build.lib.skyframe.BlacklistedPackagePrefixesValue;
import com.google.devtools.build.lib.skyframe.GraphBackedRecursivePackageProvider;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.PrepareDepsOfPatternsFunction;
import com.google.devtools.build.lib.skyframe.RecursivePackageProviderBackedTargetPatternResolver;
import com.google.devtools.build.lib.skyframe.TargetPatternValue;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.skyframe.TransitiveTraversalValue;
import com.google.devtools.build.lib.skyframe.TraversalInfoRootPackageExtractor;
import com.google.devtools.build.lib.supplier.InterruptibleSupplier;
import com.google.devtools.build.lib.supplier.MemoizingInterruptibleSupplier;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.devtools.build.skyframe.WalkableGraph.WalkableGraphFactory;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;

/**
 * {@link AbstractBlazeQueryEnvironment} that introspects the Skyframe graph to find forward and
 * reverse edges. Results obtained by calling {@link #evaluateQuery} are not guaranteed to be in any
 * particular order. As well, this class eagerly loads the full transitive closure of targets, even
 * if the full closure isn't needed.
 *
 * <p>This class has concurrent implementations of the
 * {@link QueryTaskFuture}/{@link QueryTaskCallable} helper methods. The combination of this and the
 * asynchronous evaluation model yields parallel query evaluation.
 */
public class SkyQueryEnvironment extends AbstractBlazeQueryEnvironment<Target>
    implements StreamableQueryEnvironment<Target> {
  // 10k is likely a good balance between using batch efficiently and not blowing up memory.
  // TODO(janakr): Unify with RecursivePackageProviderBackedTargetPatternResolver's constant.
  protected static final int BATCH_CALLBACK_SIZE = 10000;
  public static final int DEFAULT_THREAD_COUNT = Runtime.getRuntime().availableProcessors();
  private static final int MAX_QUERY_EXPRESSION_LOG_CHARS = 1000;
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final BlazeTargetAccessor accessor = new BlazeTargetAccessor(this);
  protected final int loadingPhaseThreads;
  protected final WalkableGraphFactory graphFactory;
  protected final ImmutableList<String> universeScope;
  protected boolean blockUniverseEvaluationErrors;
  protected ExtendedEventHandler universeEvalEventHandler;
  protected final String parserPrefix;
  protected final PathPackageLocator pkgPath;
  protected final int queryEvaluationParallelismLevel;
  private final boolean visibilityDepsAreAllowed;

  // The following fields are set in the #beforeEvaluateQuery method.
  private MultisetSemaphore<PackageIdentifier> packageSemaphore;
  protected WalkableGraph graph;
  protected InterruptibleSupplier<ImmutableSet<PathFragment>> blacklistPatternsSupplier;
  protected GraphBackedRecursivePackageProvider graphBackedRecursivePackageProvider;
  protected ListeningExecutorService executor;
  private RecursivePackageProviderBackedTargetPatternResolver resolver;
  protected final SkyKey universeKey;

  public SkyQueryEnvironment(
      boolean keepGoing,
      int loadingPhaseThreads,
      ExtendedEventHandler eventHandler,
      Set<Setting> settings,
      Iterable<QueryFunction> extraFunctions,
      String parserPrefix,
      WalkableGraphFactory graphFactory,
      List<String> universeScope,
      PathPackageLocator pkgPath,
      boolean blockUniverseEvaluationErrors) {
    this(
        keepGoing,
        loadingPhaseThreads,
        // SkyQueryEnvironment operates on a prepopulated Skyframe graph. Therefore, query
        // evaluation is completely CPU-bound.
        /*queryEvaluationParallelismLevel=*/ DEFAULT_THREAD_COUNT,
        eventHandler,
        settings,
        extraFunctions,
        parserPrefix,
        graphFactory,
        universeScope,
        pkgPath,
        blockUniverseEvaluationErrors);
  }

  protected SkyQueryEnvironment(
      boolean keepGoing,
      int loadingPhaseThreads,
      int queryEvaluationParallelismLevel,
      ExtendedEventHandler eventHandler,
      Set<Setting> settings,
      Iterable<QueryFunction> extraFunctions,
      String parserPrefix,
      WalkableGraphFactory graphFactory,
      List<String> universeScope,
      PathPackageLocator pkgPath,
      boolean blockUniverseEvaluationErrors) {
    super(
        keepGoing,
        /*strictScope=*/ true,
        /*labelFilter=*/ Rule.ALL_LABELS,
        eventHandler,
        settings,
        extraFunctions);
    this.loadingPhaseThreads = loadingPhaseThreads;
    this.graphFactory = graphFactory;
    this.pkgPath = pkgPath;
    this.universeScope = ImmutableList.copyOf(Preconditions.checkNotNull(universeScope));
    this.parserPrefix = parserPrefix;
    Preconditions.checkState(
        !universeScope.isEmpty(), "No queries can be performed with an empty universe");
    this.queryEvaluationParallelismLevel = queryEvaluationParallelismLevel;
    this.universeKey = graphFactory.getUniverseKey(universeScope, parserPrefix);
    this.blockUniverseEvaluationErrors = blockUniverseEvaluationErrors;
    this.universeEvalEventHandler =
        this.blockUniverseEvaluationErrors
            ? new ErrorBlockingForwardingEventHandler(this.eventHandler)
            : this.eventHandler;
    // In #getAllowedDeps we have special treatment of deps entailed by the `visibility` attribute.
    // Since this attribute is of the NODEP type, that means we need a special implementation of
    // NO_NODEP_DEPS.
    this.visibilityDepsAreAllowed = !settings.contains(Setting.NO_NODEP_DEPS);
  }

  @Override
  public void close() {
    if (executor != null) {
      executor.shutdownNow();
      executor = null;
    }
  }

  /** Gets roots of graph which contains all nodes needed to evaluate {@code expr}. */
  protected Set<SkyKey> getGraphRootsFromExpression(QueryExpression expr)
      throws QueryException, InterruptedException {
    return ImmutableSet.of(universeKey);
  }

  protected void beforeEvaluateQuery(QueryExpression expr)
      throws QueryException, InterruptedException {
    Set<SkyKey> roots = getGraphRootsFromExpression(expr);

    EvaluationResult<SkyValue> result;
    try (AutoProfiler p = GoogleAutoProfilerUtils.logged("evaluation and walkable graph")) {
      EvaluationContext evaluationContext =
          EvaluationContext.newBuilder()
              .setNumThreads(loadingPhaseThreads)
              .setEventHander(universeEvalEventHandler)
              .build();
      result = graphFactory.prepareAndGet(roots, configureEvaluationContext(evaluationContext));
    }

    if (graph == null || graph != result.getWalkableGraph()) {
      checkEvaluationResult(roots, result);
      packageSemaphore = makeFreshPackageMultisetSemaphore();
      graph = result.getWalkableGraph();
      blacklistPatternsSupplier = MemoizingInterruptibleSupplier.of(new BlacklistSupplier(graph));
      graphBackedRecursivePackageProvider =
          new GraphBackedRecursivePackageProvider(
              graph,
              getTargetPatternsForUniverse(),
              pkgPath,
              new TraversalInfoRootPackageExtractor());
    }

    if (executor == null) {
      executor = MoreExecutors.listeningDecorator(
          new ThreadPoolExecutor(
            /*corePoolSize=*/ queryEvaluationParallelismLevel,
            /*maximumPoolSize=*/ queryEvaluationParallelismLevel,
            /*keepAliveTime=*/ 1,
            /*units=*/ TimeUnit.SECONDS,
            /*workQueue=*/ new BlockingStack<Runnable>(),
            new ThreadFactoryBuilder().setNameFormat("QueryEnvironment %d").build()));
    }
    resolver =
        new RecursivePackageProviderBackedTargetPatternResolver(
            graphBackedRecursivePackageProvider,
            eventHandler,
            FilteringPolicies.NO_FILTER,
            packageSemaphore);
  }

  /** Returns the TargetPatterns corresponding to {@link #universeKey}. */
  protected ImmutableList<TargetPattern> getTargetPatternsForUniverse() {
    return ImmutableList.copyOf(
        Iterables.transform(
            PrepareDepsOfPatternsFunction.getTargetPatternKeys(
                PrepareDepsOfPatternsFunction.getSkyKeys(universeKey, eventHandler)),
            TargetPatternKey::getParsedPattern));
  }

  /**
   * Configures the default {@link EvaluationContext} to change the behavior of how evaluations in
   * {@link WalkableGraphFactory#prepareAndGet} work.
   */
  protected EvaluationContext configureEvaluationContext(EvaluationContext evaluationContext) {
    return evaluationContext;
  }

  protected MultisetSemaphore<PackageIdentifier> makeFreshPackageMultisetSemaphore() {
    return MultisetSemaphore.unbounded();
  }

  @ThreadSafe
  public MultisetSemaphore<PackageIdentifier> getPackageMultisetSemaphore() {
    return packageSemaphore;
  }

  boolean hasDependencyFilter() {
    return dependencyFilter != DependencyFilter.ALL_DEPS;
  }

  protected void checkEvaluationResult(Set<SkyKey> roots, EvaluationResult<SkyValue> result)
      throws QueryException {
    // If the only root is the universe key, we expect to see either a single successfully evaluated
    // value or a cycle in the result.
    if (roots.size() == 1 && Iterables.getOnlyElement(roots).equals(universeKey)) {
      Collection<SkyValue> values = result.values();
      if (!values.isEmpty()) {
        Preconditions.checkState(
            values.size() == 1,
            "Universe query \"%s\" returned multiple values unexpectedly (%s values in result)",
            universeScope,
            values.size());
        Preconditions.checkNotNull(result.get(universeKey), result);
      } else {
        // No values in the result, so there must be an error. We expect the error to be a cycle.
        boolean foundCycle = !result.getError().getCycleInfo().isEmpty();
        Preconditions.checkState(
            foundCycle,
            "Universe query \"%s\" failed with non-cycle error: %s",
            universeScope,
            result.getError());
      }
    }
  }

  private static final Duration MIN_LOGGING = Duration.ofMillis(50);

  @Override
  public final QueryExpression transformParsedQuery(QueryExpression queryExpression) {
    QueryExpressionMapper<Void> mapper = getQueryExpressionMapper();
    QueryExpression transformedQueryExpression;
    try (AutoProfiler p = GoogleAutoProfilerUtils.logged("transforming query", MIN_LOGGING)) {
      transformedQueryExpression = queryExpression.accept(mapper);
    }
    logger.atInfo().log(
        "transformed query [%s] to [%s]",
        Ascii.truncate(queryExpression.toString(), MAX_QUERY_EXPRESSION_LOG_CHARS, "[truncated]"),
        Ascii.truncate(
            transformedQueryExpression.toString(), MAX_QUERY_EXPRESSION_LOG_CHARS, "[truncated]"));
    return transformedQueryExpression;
  }

  protected QueryExpressionMapper<Void> getQueryExpressionMapper() {
    if (universeScope.size() != 1) {
      return QueryExpressionMapper.identity();
    }
    TargetPattern.Parser targetPatternParser = new TargetPattern.Parser(parserPrefix);
    String universeScopePattern = Iterables.getOnlyElement(universeScope);
    return QueryExpressionMapper.compose(
        ImmutableList.of(
            new RdepsToAllRdepsQueryExpressionMapper(targetPatternParser, universeScopePattern),
            new FilteredDirectRdepsInUniverseExpressionMapper(
                targetPatternParser, universeScopePattern)));
  }

  @Override
  protected void evalTopLevelInternal(
      QueryExpression expr, OutputFormatterCallback<Target> callback)
          throws QueryException, InterruptedException {
    Throwable throwableToThrow = null;
    try {
      super.evalTopLevelInternal(expr, callback);
    } catch (Throwable throwable) {
      throwableToThrow = throwable;
    } finally {
      if (throwableToThrow != null) {
        logger.atInfo().withCause(throwableToThrow).log(
            "About to shutdown query threadpool because of throwable");
        ListeningExecutorService obsoleteExecutor = executor;
        // Signal that executor must be recreated on the next invocation.
        executor = null;

        // If evaluation failed abruptly (e.g. was interrupted), attempt to terminate all remaining
        // tasks and then wait for them all to finish. We don't want to leave any dangling threads
        // running tasks.
        obsoleteExecutor.shutdownNow();
        boolean interrupted = false;
        boolean executorTerminated = false;
        try {
          while (!executorTerminated) {
            try {
              executorTerminated =
                  obsoleteExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
              interrupted = true;
              handleInterruptedShutdown();
            }
          }
        } finally {
          if (interrupted) {
            Thread.currentThread().interrupt();
          }
        }

        Throwables.propagateIfPossible(
            throwableToThrow, QueryException.class, InterruptedException.class);
      }
    }
  }

  /**
   * Subclasses may implement special handling when the query threadpool shutdown process is
   * interrupted. This isn't likely to happen unless there's a bug in the lifecycle management of
   * query tasks.
   */
  protected void handleInterruptedShutdown() {}

  @Override
  public QueryEvalResult evaluateQuery(
      QueryExpression expr, ThreadSafeOutputFormatterCallback<Target> callback)
          throws QueryException, InterruptedException, IOException {
    beforeEvaluateQuery(expr);

    // SkyQueryEnvironment batches callback invocations using a BatchStreamedCallback, created here
    // so that there's one per top-level evaluateQuery call. The batch size is large enough that
    // per-call costs of calling the original callback are amortized over a good number of targets,
    // and small enough that holding a batch of targets in memory doesn't risk an OOM error.
    //
    // This flushes the batched callback prior to constructing the QueryEvalResult in the unlikely
    // case of a race between the original callback and the eventHandler.
    BatchStreamedCallback batchCallback = new BatchStreamedCallback(
        callback,
        BATCH_CALLBACK_SIZE,
        createUniquifierForOuterBatchStreamedCallback(expr));
    return super.evaluateQuery(expr, batchCallback);
  }

  Map<SkyKey, Collection<Target>> targetifyValues(Map<SkyKey, ? extends Iterable<SkyKey>> input)
      throws InterruptedException {
    return targetifyValues(
        input,
        makePackageKeyToTargetKeyMap(ImmutableSet.copyOf(Iterables.concat(input.values()))));
  }

  private Map<SkyKey, Collection<Target>> targetifyValues(
      Map<SkyKey, ? extends Iterable<SkyKey>> input,
      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap) throws InterruptedException {
    ImmutableMap.Builder<SkyKey, Collection<Target>> result = ImmutableMap.builder();

    Map<SkyKey, Target> allTargets =
        getTargetKeyToTargetMapForPackageKeyToTargetKeyMap(packageKeyToTargetKeyMap);

    for (Map.Entry<SkyKey, ? extends Iterable<SkyKey>> entry : input.entrySet()) {
      Iterable<SkyKey> skyKeys = entry.getValue();
      Set<Target> targets = CompactHashSet.createWithExpectedSize(Iterables.size(skyKeys));
      for (SkyKey key : skyKeys) {
        Target target = allTargets.get(key);
        if (target != null) {
          targets.add(target);
        }
      }
      result.put(entry.getKey(), targets);
    }
    return result.build();
  }

  private Map<SkyKey, Collection<Target>> getRawReverseDeps(
      Iterable<SkyKey> transitiveTraversalKeys) throws InterruptedException {
    return targetifyValues(getReverseDepLabelsOfLabels(transitiveTraversalKeys));
  }

  protected Map<SkyKey, Iterable<SkyKey>> getReverseDepLabelsOfLabels(
      Iterable<? extends SkyKey> labels) throws InterruptedException {
    return graph.getReverseDeps(labels);
  }

  private Set<Label> getAllowedDeps(Rule rule) throws InterruptedException {
    Set<Label> allowedLabels = new HashSet<>(rule.getTransitions(dependencyFilter).values());
    if (visibilityDepsAreAllowed) {
      // Rule#getTransitions only visits the labels of attribute values, so that means it doesn't
      // know about deps from the labels of the rule's package's default_visibility. Therefore, we
      // need to explicitly handle that here.
      allowedLabels.addAll(rule.getVisibility().getDependencyLabels());
    }
    // We should add deps from aspects, otherwise they are going to be filtered out.
    allowedLabels.addAll(rule.getAspectLabelsSuperset(dependencyFilter));
    return allowedLabels;
  }

  private Collection<Target> filterFwdDeps(Target target, Collection<Target> rawFwdDeps)
      throws InterruptedException {
    if (!(target instanceof Rule)) {
      return rawFwdDeps;
    }
    final Set<Label> allowedLabels = getAllowedDeps((Rule) target);
    return Collections2.filter(rawFwdDeps,
        new Predicate<Target>() {
          @Override
          public boolean apply(Target target) {
            return allowedLabels.contains(target.getLabel());
          }
        });
  }

  @Override
  public ThreadSafeMutableSet<Target> getFwdDeps(
      Iterable<Target> targets, QueryExpressionContext<Target> context)
      throws InterruptedException {
    Map<SkyKey, Target> targetsByKey = Maps.newHashMapWithExpectedSize(Iterables.size(targets));
    for (Target target : targets) {
      targetsByKey.put(TARGET_TO_SKY_KEY.apply(target), target);
    }
    Map<SkyKey, Collection<Target>> directDeps =
        targetifyValues(getFwdDepLabels(targetsByKey.keySet()));
    if (targetsByKey.keySet().size() != directDeps.keySet().size()) {
      Iterable<SkyKey> missingTargets = Sets.difference(targetsByKey.keySet(), directDeps.keySet());
      eventHandler.handle(Event.warn("Targets were missing from graph: " + missingTargets));
    }
    ThreadSafeMutableSet<Target> result = createThreadSafeMutableSet();
    for (Map.Entry<SkyKey, Collection<Target>> entry : directDeps.entrySet()) {
      result.addAll(filterFwdDeps(targetsByKey.get(entry.getKey()), entry.getValue()));
    }
    return result;
  }

  /** Returns the target dependencies' {@link Label}s of the passed in target {@code Label}s. */
  protected Map<SkyKey, Iterable<SkyKey>> getFwdDepLabels(Iterable<SkyKey> targetLabels)
      throws InterruptedException {
    Preconditions.checkState(
        Iterables.all(targetLabels, IS_LABEL), "Expected all labels: %s", targetLabels);
    return graph.getDirectDeps(targetLabels).entrySet().stream()
        .collect(
            ImmutableMap.toImmutableMap(
                Map.Entry::getKey, entry -> Iterables.filter(entry.getValue(), IS_LABEL)));
  }

  @Override
  public QueryTaskFuture<Void> getDepsBounded(
      QueryExpression queryExpression,
      QueryExpressionContext<Target> context,
      Callback<Target> callback,
      int depthBound,
      QueryExpression caller) {
    // Re-implement the bounded deps algorithm to allow for proper error reporting of missing
    // targets that cannot be targetified.
    final MinDepthUniquifier<Target> minDepthUniquifier = createMinDepthUniquifier();
    return eval(
        queryExpression,
        context,
        partialResult -> {
          ThreadSafeMutableSet<Target> current = createThreadSafeMutableSet();
          Iterables.addAll(current, partialResult);

          for (int i = 0; i <= depthBound; i++) {
            // Filter already visited nodes: if we see a node in a later round, then we don't need
            // to visit it again, because the depth at which we see it at must be greater than or
            // equal to the last visit.
            ImmutableList<Target> toProcess =
                minDepthUniquifier.uniqueAtDepthLessThanOrEqualTo(current, i);
            callback.process(toProcess);

            if (i == depthBound) {
              // We don't need to fetch dep targets any more.
              break;
            }

            current = createThreadSafeMutableSet();
            addTargetsOfDirectDepsAndReportErrorsIfAny(context, caller, current, toProcess);

            if (current.isEmpty()) {
              // Exit when there are no more nodes to visit.
              break;
            }
          }
        });
  }

  protected void addTargetsOfDirectDepsAndReportErrorsIfAny(
      QueryExpressionContext<Target> context,
      QueryExpression caller,
      ThreadSafeMutableSet<Target> toAddTo,
      ImmutableList<Target> toProcess)
      throws InterruptedException, QueryException {
    Map<SkyKey, Iterable<SkyKey>> keyToDepKeys =
        getFwdDepLabels(toProcess.stream().map(Target::getLabel).collect(Collectors.toList()));
    Map<SkyKey, Collection<Target>> targetDepMap = targetifyValues(keyToDepKeys);

    Map<SkyKey, Target> targetMap = new HashMap<>();
    Set<SkyKey> depLabels = ImmutableSet.copyOf(Iterables.concat(keyToDepKeys.values()));
    for (Collection<Target> depTargets : targetDepMap.values()) {
      for (Target depTarget : depTargets) {
        targetMap.putIfAbsent(depTarget.getLabel(), depTarget);
        toAddTo.add(depTarget);
      }
    }
    reportUnsuccessfulOrMissingTargets(targetMap, depLabels, caller);
  }

  @Override
  public Collection<Target> getReverseDeps(
      Iterable<Target> targets, QueryExpressionContext<Target> context)
      throws InterruptedException {
    return processRawReverseDeps(
        getReverseDepsOfLabels(Iterables.transform(targets, Target::getLabel)));
  }

  protected Map<SkyKey, Collection<Target>> getReverseDepsOfLabels(Iterable<Label> targetLabels)
      throws InterruptedException {
    return getRawReverseDeps(Iterables.transform(targetLabels, label -> label));
  }

  /** Targetify SkyKeys of reverse deps and filter out targets whose deps are not allowed. */
  Collection<Target> filterRawReverseDepsOfTransitiveTraversalKeys(
      Map<SkyKey, ? extends Iterable<SkyKey>> rawReverseDeps,
      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap) throws InterruptedException {
    return processRawReverseDeps(targetifyValues(rawReverseDeps, packageKeyToTargetKeyMap));
  }

  private Collection<Target> processRawReverseDeps(Map<SkyKey, Collection<Target>> rawReverseDeps)
      throws InterruptedException {
    Set<Target> result = CompactHashSet.create();
    CompactHashSet<Target> visited =
        CompactHashSet.createWithExpectedSize(totalSizeOfCollections(rawReverseDeps.values()));

    Set<Label> keys = CompactHashSet.create(Collections2.transform(rawReverseDeps.keySet(),
        SKYKEY_TO_LABEL));
    for (Collection<Target> parentCollection : rawReverseDeps.values()) {
      for (Target parent : parentCollection) {
        if (visited.add(parent)) {
          if (parent instanceof Rule && dependencyFilter != DependencyFilter.ALL_DEPS) {
            for (Label label : getAllowedDeps((Rule) parent)) {
              if (keys.contains(label)) {
                result.add(parent);
              }
            }
          } else {
            result.add(parent);
          }
        }
      }
    }
    return result;
  }

  private static <T> int totalSizeOfCollections(Iterable<Collection<T>> nestedCollections) {
    int totalSize = 0;
    for (Collection<T> collection : nestedCollections) {
      totalSize += collection.size();
    }
    return totalSize;
  }

  @Override
  public ThreadSafeMutableSet<Target> getTransitiveClosure(
      ThreadSafeMutableSet<Target> targets, QueryExpressionContext<Target> context)
      throws InterruptedException {
    return SkyQueryUtils.getTransitiveClosure(
        targets, targets1 -> getFwdDeps(targets1, context), createThreadSafeMutableSet());
  }

  @Override
  public ImmutableList<Target> getNodesOnPath(
      Target from, Target to, QueryExpressionContext<Target> context) throws InterruptedException {
    return SkyQueryUtils.getNodesOnPath(
        from, to, targets -> getFwdDeps(targets, context), Target::getLabel);
  }

  protected final <R> ListenableFuture<R> safeSubmit(Callable<R> callable) {
    try {
      return executor.submit(callable);
    } catch (RejectedExecutionException e) {
      return Futures.immediateCancelledFuture();
    }
  }

  @SuppressWarnings("unchecked")
  private <R> ListenableFuture<R> safeSubmitAsync(QueryTaskAsyncCallable<R> callable) {
    try {
      return Futures.submitAsync(() -> (ListenableFuture<R>) callable.call(), executor);
    } catch (RejectedExecutionException e) {
      return Futures.immediateCancelledFuture();
    }
  }

  @ThreadSafe
  @Override
  public QueryTaskFuture<Void> eval(
      final QueryExpression expr,
      final QueryExpressionContext<Target> context,
      final Callback<Target> callback) {
    // TODO(bazel-team): As in here, use concurrency for the async #eval of other QueryEnvironment
    // implementations.
    return executeAsync(() -> expr.eval(SkyQueryEnvironment.this, context, callback));
  }

  @Override
  public <R> QueryTaskFuture<R> execute(QueryTaskCallable<R> callable) {
    return QueryTaskFutureImpl.ofDelegate(safeSubmit(callable));
  }

  @Override
  public <R> QueryTaskFuture<R> executeAsync(QueryTaskAsyncCallable<R> callable) {
    return QueryTaskFutureImpl.ofDelegate(safeSubmitAsync(callable));
  }

  @Override
  public <T1, T2> QueryTaskFuture<T2> transformAsync(
      QueryTaskFuture<T1> future,
      final Function<T1, QueryTaskFuture<T2>> function) {
    return QueryTaskFutureImpl.ofDelegate(
        Futures.transformAsync(
            (QueryTaskFutureImpl<T1>) future,
            input -> (QueryTaskFutureImpl<T2>) function.apply(input),
            executor));
  }

  @Override
  public <R> QueryTaskFuture<R> whenAllSucceedCall(
      Iterable<? extends QueryTaskFuture<?>> futures, QueryTaskCallable<R> callable) {
    return QueryTaskFutureImpl.ofDelegate(
        Futures.whenAllSucceed(cast(futures)).call(callable, executor));
  }

  @ThreadSafe
  @Override
  public ThreadSafeMutableSet<Target> createThreadSafeMutableSet() {
    return new ThreadSafeMutableKeyExtractorBackedSetImpl<>(
        TargetKeyExtractor.INSTANCE, Target.class, queryEvaluationParallelismLevel);
  }

  @Override
  public <V> MutableMap<Target, V> createMutableMap() {
    return new MutableKeyExtractorBackedMapImpl<>(TargetKeyExtractor.INSTANCE);
  }

  @ThreadSafe
  protected NonExceptionalUniquifier<Target> createUniquifierForOuterBatchStreamedCallback(
      QueryExpression expr) {
    return createUniquifier();
  }

  @ThreadSafe
  @Override
  public NonExceptionalUniquifier<Target> createUniquifier() {
    return new UniquifierImpl<>(TargetKeyExtractor.INSTANCE, queryEvaluationParallelismLevel);
  }

  @ThreadSafe
  @Override
  public MinDepthUniquifier<Target> createMinDepthUniquifier() {
    return new MinDepthUniquifierImpl<>(
        TargetKeyExtractor.INSTANCE, queryEvaluationParallelismLevel);
  }

  @ThreadSafe
  public MinDepthUniquifier<SkyKey> createMinDepthSkyKeyUniquifier() {
    return new MinDepthUniquifierImpl<>(
        SkyKeyKeyExtractor.INSTANCE, queryEvaluationParallelismLevel);
  }

  @ThreadSafe
  public Uniquifier<SkyKey> createSkyKeyUniquifier() {
    return new UniquifierImpl<>(SkyKeyKeyExtractor.INSTANCE, queryEvaluationParallelismLevel);
  }

  private ImmutableSet<PathFragment> getBlacklistedExcludes(TargetPatternKey targetPatternKey)
  throws InterruptedException {
    return targetPatternKey.getAllBlacklistedSubdirectoriesToExclude(blacklistPatternsSupplier);
  }

  @ThreadSafe
  @Override
  public Collection<Target> getSiblingTargetsInPackage(Target target) {
    return target.getPackage().getTargets().values();
  }

  @ThreadSafe
  @Override
  public QueryTaskFuture<Void> getTargetsMatchingPattern(
      QueryExpression owner, String pattern, Callback<Target> callback) {
    TargetPatternKey targetPatternKey;
    try {
      targetPatternKey = TargetPatternValue.key(pattern, FilteringPolicies.NO_FILTER, parserPrefix);
    } catch (TargetParsingException tpe) {
      try {
        reportBuildFileError(owner, tpe.getMessage());
      } catch (QueryException qe) {
        return immediateFailedFuture(qe);
      }
      return immediateSuccessfulFuture(null);
    }
    return evalTargetPatternKey(owner, targetPatternKey, callback);
  }

  @ThreadSafe
  public QueryTaskFuture<Void> evalTargetPatternKey(
      QueryExpression owner, TargetPatternKey targetPatternKey, Callback<Target> callback) {
    ImmutableSet<PathFragment> blacklistedSubdirectoriesToExclude;
    try {
      blacklistedSubdirectoriesToExclude = getBlacklistedExcludes(targetPatternKey);
    } catch (InterruptedException ie) {
      return immediateCancelledFuture();
    }
    TargetPattern patternToEval = targetPatternKey.getParsedPattern();
    ImmutableSet<PathFragment> additionalSubdirectoriesToExclude =
        targetPatternKey.getExcludedSubdirectories();
    AsyncFunction<TargetParsingException, Void> reportBuildFileErrorAsyncFunction =
        exn -> {
          reportBuildFileError(owner, exn.getMessage());
          return Futures.immediateFuture(null);
        };
    Callback<Target> filteredCallback = callback;
    if (!targetPatternKey.getPolicy().equals(NO_FILTER)) {
      filteredCallback =
          targets ->
              callback.process(
                  Iterables.filter(
                      targets,
                      target ->
                          targetPatternKey.getPolicy().shouldRetain(target, /*explicit=*/ false)));
    }
    ListenableFuture<Void> evalFuture =
        patternToEval.evalAsync(
            resolver,
            blacklistedSubdirectoriesToExclude,
            additionalSubdirectoriesToExclude,
            filteredCallback,
            QueryException.class,
            executor);
    return QueryTaskFutureImpl.ofDelegate(
        Futures.catchingAsync(
            evalFuture,
            TargetParsingException.class,
            reportBuildFileErrorAsyncFunction,
            directExecutor()));
  }

  @ThreadSafe
  @Override
  public ThreadSafeMutableSet<Target> getBuildFiles(
      QueryExpression caller,
      ThreadSafeMutableSet<Target> nodes,
      boolean buildFiles,
      boolean loads,
      QueryExpressionContext<Target> context)
      throws QueryException, InterruptedException {
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
          addIfUniqueLabel(pkg.getBuildFile(), seenLabels, dependentFiles);
        }

        List<Label> extensions = new ArrayList<>();
        if (loads) {
          extensions.addAll(pkg.getStarlarkFileDependencies());
        }

        for (Label extension : extensions) {

          Target loadTarget = getLoadTarget(extension, pkg);
          addIfUniqueLabel(loadTarget, seenLabels, dependentFiles);

          // Also add the BUILD file of the extension.
          if (buildFiles) {
            Label buildFileLabel = getBuildFileLabel(loadTarget.getLabel().getPackageIdentifier());
            addIfUniqueLabel(new FakeLoadTarget(buildFileLabel, pkg), seenLabels, dependentFiles);
          }
        }
      }
    }
    return dependentFiles;
  }

  protected Label getBuildFileLabel(PackageIdentifier packageIdentifier) throws QueryException {
    // TODO(bazel-team): Try avoid filesystem access here.
    Path buildFileForLoad = null;
    try {
      buildFileForLoad = pkgPath.getPackageBuildFile(packageIdentifier);
    } catch (NoSuchPackageException e) {
      throw new QueryException(packageIdentifier + " does not exist in graph");
    }
    return Label.createUnvalidated(packageIdentifier, buildFileForLoad.getBaseName());
  }

  private static void addIfUniqueLabel(Target node, Set<Label> labels, Set<Target> nodes) {
    if (labels.add(node.getLabel())) {
      nodes.add(node);
    }
  }

  protected int getVisitBatchSizeForParallelVisitation() {
    return ParallelSkyQueryUtils.VISIT_BATCH_SIZE;
  }

  public VisitTaskStatusCallback getVisitTaskStatusCallback() {
    return VisitTaskStatusCallback.NULL_INSTANCE;
  }

  private Target getLoadTarget(Label label, Package pkg) {
    return new FakeLoadTarget(label, pkg);
  }

  int getQueryEvaluationParallelismLevel() {
    return queryEvaluationParallelismLevel;
  }

  @ThreadSafe
  @Override
  public TargetAccessor<Target> getAccessor() {
    return accessor;
  }

  @ThreadSafe
  private Package getPackage(PackageIdentifier packageIdentifier)
      throws InterruptedException, QueryException, NoSuchPackageException {
    SkyKey packageKey = PackageValue.key(packageIdentifier);
    PackageValue packageValue = (PackageValue) graph.getValue(packageKey);
    if (packageValue != null) {
      Package pkg = packageValue.getPackage();
      if (pkg.containsErrors()) {
        throw new BuildFileContainsErrorsException(packageIdentifier);
      }
      return pkg;
    } else {
      NoSuchPackageException exception = (NoSuchPackageException) graph.getException(packageKey);
      if (exception != null) {
        throw exception;
      }
      if (graph.isCycle(packageKey)) {
        throw new NoSuchPackageException(packageIdentifier, "Package depends on a cycle");
      } else {
        throw new QueryException(packageKey + " does not exist in graph");
      }
    }
  }

  @ThreadSafe
  @Override
  public Target getTarget(Label label)
      throws TargetNotFoundException, QueryException, InterruptedException {
    try {
      Package pkg = getPackage(label.getPackageIdentifier());
      return pkg.getTarget(label.getName());
    } catch (NoSuchThingException e) {
      throw new TargetNotFoundException(e);
    }
  }

  @Override
  public Map<Label, Target> getTargets(Iterable<Label> labels) throws InterruptedException {
    if (Iterables.isEmpty(labels)) {
      return ImmutableMap.of();
    }
    Multimap<PackageIdentifier, Label> packageIdToLabelMap = ArrayListMultimap.create();
    labels.forEach(label -> packageIdToLabelMap.put(label.getPackageIdentifier(), label));
    Map<PackageIdentifier, Package> packageIdToPackageMap =
        bulkGetPackages(packageIdToLabelMap.keySet());
    ImmutableMap.Builder<Label, Target> resultBuilder = ImmutableMap.builder();
    packageSemaphore.acquireAll(packageIdToLabelMap.keySet());
    try {
      for (PackageIdentifier pkgId : packageIdToLabelMap.keySet()) {
        Package pkg = packageIdToPackageMap.get(pkgId);
        if (pkg == null) {
          continue;
        }
        for (Label label : packageIdToLabelMap.get(pkgId)) {
          Target target;
          try {
            target = pkg.getTarget(label.getName());
          } catch (NoSuchTargetException e) {
            continue;
          }
          resultBuilder.put(label, target);
        }
      }
      return resultBuilder.build();
    } finally {
      packageSemaphore.releaseAll(packageIdToLabelMap.keySet());
    }
  }

  @ThreadSafe
  public Map<PackageIdentifier, Package> bulkGetPackages(Iterable<PackageIdentifier> pkgIds)
      throws InterruptedException {
    Set<SkyKey> pkgKeys = ImmutableSet.copyOf(PackageValue.keys(pkgIds));
    ImmutableMap.Builder<PackageIdentifier, Package> pkgResults = ImmutableMap.builder();
    Map<SkyKey, SkyValue> packages = graph.getSuccessfulValues(pkgKeys);
    for (Map.Entry<SkyKey, SkyValue> pkgEntry : packages.entrySet()) {
      PackageIdentifier pkgId = (PackageIdentifier) pkgEntry.getKey().argument();
      PackageValue pkgValue = (PackageValue) pkgEntry.getValue();
      pkgResults.put(pkgId, Preconditions.checkNotNull(pkgValue.getPackage(), pkgId));
    }
    return pkgResults.build();
  }

  @Override
  public void buildTransitiveClosure(
      QueryExpression caller,
      ThreadSafeMutableSet<Target> targets,
      int maxDepth) throws QueryException, InterruptedException {
    // Everything has already been loaded, so here we just check for errors so that we can
    // pre-emptively throw/report if needed.
    reportUnsuccessfulOrMissingTargetsInternal(targets, ImmutableSet.of(), caller);
  }

  @Override
  protected void preloadOrThrow(QueryExpression caller, Collection<String> patterns)
      throws QueryException, TargetParsingException {
    // SkyQueryEnvironment directly evaluates target patterns in #getTarget and similar methods
    // using its graph, which is prepopulated using the universeScope (see #beforeEvaluateQuery),
    // so no preloading of target patterns is necessary.
  }

  public void reportUnsuccessfulOrMissingTargets(
      Map<? extends SkyKey, Target> keysWithTargets,
      Set<SkyKey> allTargetKeys,
      QueryExpression caller)
      throws InterruptedException, QueryException {
    Set<SkyKey> missingTargets = new HashSet<>();
    Set<? extends SkyKey> keysFound = keysWithTargets.keySet();
    for (SkyKey key : allTargetKeys) {
      if (!keysFound.contains(key)) {
        missingTargets.add(key);
      }
    }
    reportUnsuccessfulOrMissingTargetsInternal(keysWithTargets.values(), missingTargets, caller);
  }

  private void reportUnsuccessfulOrMissingTargetsInternal(
      Iterable<Target> targets, Iterable<SkyKey> missingTargetKeys, QueryExpression caller)
      throws InterruptedException, QueryException {
    // Targets can be in four states:
    //  (1) Existent TransitiveTraversalValue with no error
    //  (2) Existent TransitiveTraversalValue with an error (eg. transitive dependency error)
    //  (3) Non-existent because it threw a SkyFunctionException
    //  (4) Non-existent because it was never evaluated
    //
    // We first find the errors in the existent TransitiveTraversalValues that have an error. We
    // then find which keys correspond to SkyFunctionExceptions and extract the errors from those.
    // Lastly, any leftover keys are marked as missing from the graph and an error is produced.
    ImmutableList.Builder<String> errorMessagesBuilder = ImmutableList.builder();
    Set<SkyKey> successfulKeys = filterSuccessfullyLoadedTargets(targets, errorMessagesBuilder);

    Iterable<SkyKey> keysWithTarget = makeLabels(targets);
    // Next, look for errors from the unsuccessfully evaluated TransitiveTraversal skyfunctions.
    Iterable<SkyKey> unsuccessfulKeys =
        Iterables.filter(keysWithTarget, Predicates.not(Predicates.in(successfulKeys)));
    Iterable<SkyKey> unsuccessfulOrMissingKeys =
        Iterables.concat(unsuccessfulKeys, missingTargetKeys);
    processUnsuccessfulAndMissingKeys(unsuccessfulOrMissingKeys, errorMessagesBuilder);

    // Lastly, report all found errors.
    if (!Iterables.isEmpty(unsuccessfulOrMissingKeys)) {
      eventHandler.handle(
          Event.warn("Targets were missing from graph: " + unsuccessfulOrMissingKeys));
    }
    for (String errorMessage : errorMessagesBuilder.build()) {
      reportBuildFileError(caller, errorMessage);
    }
  }

  // Finds labels that were evaluated but resulted in an exception, adding the error message to the
  // passed-in errorMessagesBuilder.
  protected void processUnsuccessfulAndMissingKeys(
      Iterable<SkyKey> unsuccessfulKeys, ImmutableList.Builder<String> errorMessagesBuilder)
      throws InterruptedException {
    Set<Map.Entry<SkyKey, Exception>> errorEntries =
        graph.getMissingAndExceptions(unsuccessfulKeys).entrySet();
    for (Map.Entry<SkyKey, Exception> entry : errorEntries) {
      if (entry.getValue() != null) {
        errorMessagesBuilder.add(entry.getValue().getMessage());
      }
    }
  }

  // Filters for successful targets while storing error messages of unsuccessful targets.
  protected Set<SkyKey> filterSuccessfullyLoadedTargets(
      Iterable<Target> targets, ImmutableList.Builder<String> errorMessagesBuilder)
      throws InterruptedException {
    Iterable<SkyKey> transitiveTraversalKeys = makeLabels(targets);

    // First, look for errors in the successfully evaluated TransitiveTraversalValues. They may
    // have encountered errors that they were able to recover from.
    Set<Map.Entry<SkyKey, SkyValue>> successfulEntries =
        graph.getSuccessfulValues(transitiveTraversalKeys).entrySet();
    ImmutableSet.Builder<SkyKey> successfulKeysBuilder = ImmutableSet.builder();
    for (Map.Entry<SkyKey, SkyValue> successfulEntry : successfulEntries) {
      successfulKeysBuilder.add(successfulEntry.getKey());
      TransitiveTraversalValue value = (TransitiveTraversalValue) successfulEntry.getValue();
      String errorMessage = value.getErrorMessage();
      if (errorMessage != null) {
        errorMessagesBuilder.add(errorMessage);
      }
    }
    return successfulKeysBuilder.build();
  }

  public ExtendedEventHandler getEventHandler() {
    return eventHandler;
  }

  public static final Predicate<SkyKey> IS_LABEL =
      SkyFunctionName.functionIs(Label.TRANSITIVE_TRAVERSAL);

  public static final Function<SkyKey, Label> SKYKEY_TO_LABEL =
      skyKey -> IS_LABEL.apply(skyKey) ? (Label) skyKey.argument() : null;

  static final Function<SkyKey, PackageIdentifier> PACKAGE_SKYKEY_TO_PACKAGE_IDENTIFIER =
      skyKey -> (PackageIdentifier) skyKey.argument();

  public static Multimap<SkyKey, SkyKey> makePackageKeyToTargetKeyMap(Iterable<SkyKey> keys) {
    Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap = ArrayListMultimap.create();
    for (SkyKey key : keys) {
      Label label = SKYKEY_TO_LABEL.apply(key);
      if (label == null) {
        continue;
      }
      packageKeyToTargetKeyMap.put(PackageValue.key(label.getPackageIdentifier()), key);
    }
    return packageKeyToTargetKeyMap;
  }

  public static Set<PackageIdentifier> getPkgIdsNeededForTargetification(
      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap) {
    return packageKeyToTargetKeyMap
        .keySet()
        .stream()
        .map(SkyQueryEnvironment.PACKAGE_SKYKEY_TO_PACKAGE_IDENTIFIER)
        .collect(toImmutableSet());
  }

  @ThreadSafe
  public Map<SkyKey, Target> getTargetKeyToTargetMapForPackageKeyToTargetKeyMap(
      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap) throws InterruptedException {
    ImmutableMap.Builder<SkyKey, Target> resultBuilder = ImmutableMap.builder();
    getTargetsForPackageKeyToTargetKeyMapHelper(packageKeyToTargetKeyMap, resultBuilder::put);
    return resultBuilder.build();
  }

  @ThreadSafe
  public Multimap<PackageIdentifier, Target> getPkgIdToTargetMultimapForPackageKeyToTargetKeyMap(
      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap) throws InterruptedException {
    Multimap<PackageIdentifier, Target> result = ArrayListMultimap.create();
    getTargetsForPackageKeyToTargetKeyMapHelper(
        packageKeyToTargetKeyMap,
        (k, t) -> result.put(t.getLabel().getPackageIdentifier(), t));
    return result;
  }

  private void getTargetsForPackageKeyToTargetKeyMapHelper(
      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap,
      BiConsumer<SkyKey, Target> targetKeyAndTargetConsumer) throws InterruptedException {
    Set<SkyKey> processedTargets = new HashSet<>();
    Map<SkyKey, SkyValue> packageMap = graph.getSuccessfulValues(packageKeyToTargetKeyMap.keySet());
    for (Map.Entry<SkyKey, SkyValue> entry : packageMap.entrySet()) {
      Package pkg = ((PackageValue) entry.getValue()).getPackage();
      for (SkyKey targetKey : packageKeyToTargetKeyMap.get(entry.getKey())) {
        if (processedTargets.add(targetKey)) {
          try {
            Target target = pkg.getTarget(SKYKEY_TO_LABEL.apply(targetKey).getName());
            targetKeyAndTargetConsumer.accept(targetKey, target);
          } catch (NoSuchTargetException e) {
            // Skip missing target.
          }
        }
      }
    }
  }

  static final Function<Target, SkyKey> TARGET_TO_SKY_KEY =
      target -> TransitiveTraversalValue.key(target.getLabel());

  /** A strict (i.e. non-lazy) variant of {@link #makeLabels}. */
  public static <T extends Target> Iterable<SkyKey> makeLabelsStrict(Iterable<T> targets) {
    return ImmutableList.copyOf(makeLabels(targets));
  }

  protected static <T extends Target> Iterable<SkyKey> makeLabels(Iterable<T> targets) {
    return Iterables.transform(targets, TARGET_TO_SKY_KEY);
  }

  @Override
  public Target getOrCreate(Target target) {
    return target;
  }

  protected Iterable<Target> getBuildFileTargetsForPackageKeys(
      Set<PackageIdentifier> pkgIds, QueryExpressionContext<Target> context)
      throws QueryException, InterruptedException {
    packageSemaphore.acquireAll(pkgIds);
    try {
      return Iterables.transform(
          graph.getSuccessfulValues(PackageValue.keys(pkgIds)).values(),
          skyValue -> ((PackageValue) skyValue).getPackage().getBuildFile());
    } finally {
      packageSemaphore.releaseAll(pkgIds);
    }
  }

  /**
   * Calculates the set of packages whose evaluation transitively depends on (e.g. via 'load'
   * statements) the contents of the specified paths. The emitted {@link Target}s are BUILD file
   * targets.
   */
  @ThreadSafe
  QueryTaskFuture<Void> getRBuildFiles(
      Collection<PathFragment> fileIdentifiers,
      QueryExpressionContext<Target> context,
      Callback<Target> callback) {
    return QueryTaskFutureImpl.ofDelegate(
        safeSubmit(
            () -> {
              ParallelSkyQueryUtils.getRBuildFilesParallel(
                  SkyQueryEnvironment.this, fileIdentifiers, context, callback);
              return null;
            }));
  }

  @Override
  public Iterable<QueryFunction> getFunctions() {
    return ImmutableList.<QueryFunction>builder()
        .addAll(super.getFunctions())
        .add(new AllRdepsFunction())
        .add(new RBuildFilesFunction())
        .build();
  }

  private static class BlacklistSupplier
      implements InterruptibleSupplier<ImmutableSet<PathFragment>> {
    private final WalkableGraph graph;

    private BlacklistSupplier(WalkableGraph graph) {
      this.graph = graph;
    }

    @Override
    public ImmutableSet<PathFragment> get() throws InterruptedException {
      return ((BlacklistedPackagePrefixesValue)
              graph.getValue(BlacklistedPackagePrefixesValue.key()))
          .getPatterns();
    }
  }

  private static class SkyKeyKeyExtractor implements KeyExtractor<SkyKey, SkyKey> {
    private static final SkyKeyKeyExtractor INSTANCE = new SkyKeyKeyExtractor();

    private SkyKeyKeyExtractor() {
    }

    @Override
    public SkyKey extractKey(SkyKey element) {
      return element;
    }
  }

  /**
   * Wraps a {@link Callback} and guarantees that all calls to the original will have at least
   * {@code batchThreshold} {@link Target}s, except for the final such call.
   *
   * <p>Retains fewer than {@code batchThreshold} {@link Target}s at a time.
   *
   * <p>After this object's {@link #process} has been called for the last time, {#link
   * #processLastPending} must be called to "flush" any remaining {@link Target}s through to the
   * original.
   *
   * <p>This callback may be called from multiple threads concurrently. At most one thread will call
   * the wrapped {@code callback} concurrently.
   */
  // TODO(nharmata): For queries with less than {@code batchThreshold} results, this batching
  // strategy probably hurts performance since we can only start formatting results once the entire
  // query is finished.
  // TODO(nharmata): This batching strategy is also potentially harmful from a memory perspective
  // since when the Targets being output are backed by Package instances, we're delaying GC of the
  // Package instances until the output batch size is met.
  private static class BatchStreamedCallback extends ThreadSafeOutputFormatterCallback<Target>
      implements Callback<Target> {

    // TODO(nharmata): Now that we know the wrapped callback is ThreadSafe, there's no correctness
    // concern that requires the prohibition of concurrent uses of the callback; the only concern is
    // memory. We should have a threshold for when to invoke the callback with a batch, and also a
    // separate, larger, bound on the number of targets being processed at the same time.
    private final ThreadSafeOutputFormatterCallback<Target> callback;
    private final NonExceptionalUniquifier<Target> uniquifier;
    private final Object pendingLock = new Object();
    private List<Target> pending = new ArrayList<>();
    private int batchThreshold;

    private BatchStreamedCallback(
        ThreadSafeOutputFormatterCallback<Target> callback,
        int batchThreshold,
        NonExceptionalUniquifier<Target> uniquifier) {
      this.callback = callback;
      this.batchThreshold = batchThreshold;
      this.uniquifier = uniquifier;
    }

    @Override
    public void start() throws IOException {
      callback.start();
    }

    @Override
    public void processOutput(Iterable<Target> partialResult)
        throws IOException, InterruptedException {
      ImmutableList<Target> uniquifiedTargets = uniquifier.unique(partialResult);
      Iterable<Target> toProcess = null;
      synchronized (pendingLock) {
        Preconditions.checkNotNull(pending, "Reuse of the callback is not allowed");
        pending.addAll(uniquifiedTargets);
        if (pending.size() >= batchThreshold) {
          toProcess = pending;
          pending = new ArrayList<>();
        }
      }
      if (toProcess != null) {
        callback.processOutput(toProcess);
      }
    }

    @Override
    public void close(boolean failFast) throws IOException, InterruptedException {
      if (!failFast) {
        processLastPending();
      }
      callback.close(failFast);
    }

    private void processLastPending() throws IOException, InterruptedException {
      synchronized (pendingLock) {
        if (!pending.isEmpty()) {
          callback.processOutput(pending);
          pending = null;
        }
      }
    }
  }

  @ThreadSafe
  @Override
  public QueryTaskFuture<Void> getAllRdepsUnboundedParallel(
      QueryExpression expression,
      QueryExpressionContext<Target> context,
      Callback<Target> callback) {
    return ParallelSkyQueryUtils.getAllRdepsUnboundedParallel(this, expression, context, callback);
  }

  @ThreadSafe
  @Override
  public QueryTaskFuture<Void> getAllRdepsBoundedParallel(
      QueryExpression expression,
      int depth,
      QueryExpressionContext<Target> context,
      Callback<Target> callback) {
    return ParallelSkyQueryUtils.getAllRdepsBoundedParallel(
        this, expression, depth, context, callback);
  }

  protected QueryTaskFuture<Predicate<SkyKey>> getUnfilteredUniverseDTCSkyKeyPredicateFuture(
      QueryExpression universe, QueryExpressionContext<Target> context) {
    return ParallelSkyQueryUtils.getDTCSkyKeyPredicateFuture(
        this,
        universe,
        context,
        BATCH_CALLBACK_SIZE,
        queryEvaluationParallelismLevel);
  }

  @ThreadSafe
  @Override
  public QueryTaskFuture<Void> getRdepsUnboundedParallel(
      QueryExpression expression,
      QueryExpression universe,
      QueryExpressionContext<Target> context,
      Callback<Target> callback) {
    return transformAsync(
        // Even if we need to do edge filtering, it's fine to construct the rdeps universe via an
        // unfiltered DTC visitation; the subsequent rdeps visitation will perform the edge
        // filtering.
        getUnfilteredUniverseDTCSkyKeyPredicateFuture(universe, context),
        unfilteredUniversePredicate ->
            ParallelSkyQueryUtils.getRdepsInUniverseUnboundedParallel(
                this, expression, unfilteredUniversePredicate, context, callback));
  }

  @Override
  public QueryTaskFuture<Void> getDepsUnboundedParallel(
      QueryExpression expression,
      QueryExpressionContext<Target> context,
      Callback<Target> callback,
      QueryExpression caller) {
    return ParallelSkyQueryUtils.getDepsUnboundedParallel(
        SkyQueryEnvironment.this,
        expression,
        context,
        callback,
        /*depsNeedFiltering=*/ !dependencyFilter.equals(DependencyFilter.ALL_DEPS),
        caller);
  }

  @ThreadSafe
  @Override
  public QueryTaskFuture<Void> getRdepsBoundedParallel(
      QueryExpression expression,
      int depth,
      QueryExpression universe,
      QueryExpressionContext<Target> context,
      Callback<Target> callback) {
    return transformAsync(
        // Even if we need to do edge filtering, it's fine to construct the rdeps universe via an
        // unfiltered DTC visitation; the subsequent rdeps visitation will perform the edge
        // filtering.
        getUnfilteredUniverseDTCSkyKeyPredicateFuture(universe, context),
        universePredicate ->
            ParallelSkyQueryUtils.getRdepsInUniverseBoundedParallel(
                this, expression, depth, universePredicate, context, callback));
  }

  /**
   * Query evaluation behavior is specified with respect to errors it emits. (Or at least it should
   * be. Tools rely on it.) Notably, errors that occur during evaluation of a query's universe must
   * not be emitted during query command evaluation. Consider the case of a simple single target
   * query when {@code //...} is the universe: errors in far flung parts of the workspace should not
   * be emitted when that query command is evaluated.
   *
   * <p>Non-error message events are not specified. For instance, it's useful (and expected by some
   * unit tests that should know better) for query commands to emit {@link EventKind#PROGRESS}
   * events during package loading.
   *
   * <p>Therefore, this class is used to forward only non-{@link EventKind#ERROR} events during
   * universe loading to the {@link SkyQueryEnvironment}'s {@link ExtendedEventHandler}.
   */
  protected static class ErrorBlockingForwardingEventHandler extends DelegatingEventHandler {

    public ErrorBlockingForwardingEventHandler(ExtendedEventHandler delegate) {
      super(delegate);
    }

    @Override
    public void handle(Event e) {
      if (!e.getKind().equals(EventKind.ERROR)) {
        super.handle(e);
      }
    }
  }
}
