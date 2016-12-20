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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Throwables;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSet.Builder;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.collect.CompactHashSet;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.concurrent.NamedForkJoinPool;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.query2.engine.AllRdepsFunction;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.FunctionExpression;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryExpressionEvalListener;
import com.google.devtools.build.lib.query2.engine.QueryExpressionMapper;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AbstractThreadSafeUniquifier;
import com.google.devtools.build.lib.query2.engine.RdepsFunction;
import com.google.devtools.build.lib.query2.engine.StreamableQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.TargetLiteral;
import com.google.devtools.build.lib.query2.engine.ThreadSafeCallback;
import com.google.devtools.build.lib.query2.engine.ThreadSafeUniquifier;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.lib.query2.engine.VariableContext;
import com.google.devtools.build.lib.skyframe.BlacklistedPackagePrefixesValue;
import com.google.devtools.build.lib.skyframe.ContainingPackageLookupFunction;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.GraphBackedRecursivePackageProvider;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.PrepareDepsOfPatternsFunction;
import com.google.devtools.build.lib.skyframe.RecursivePackageProviderBackedTargetPatternResolver;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.TargetPatternValue;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.skyframe.TransitiveTraversalValue;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InterruptibleSupplier;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.devtools.build.skyframe.WalkableGraph.WalkableGraphFactory;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * {@link AbstractBlazeQueryEnvironment} that introspects the Skyframe graph to find forward and
 * reverse edges. Results obtained by calling {@link #evaluateQuery} are not guaranteed to be in any
 * particular order. As well, this class eagerly loads the full transitive closure of targets, even
 * if the full closure isn't needed.
 */
public class SkyQueryEnvironment extends AbstractBlazeQueryEnvironment<Target>
    implements StreamableQueryEnvironment<Target> {
  // 10k is likely a good balance between using batch efficiently and not blowing up memory.
  // TODO(janakr): Unify with RecursivePackageProviderBackedTargetPatternResolver's constant.
  static final int BATCH_CALLBACK_SIZE = 10000;
  protected static final int DEFAULT_THREAD_COUNT = Runtime.getRuntime().availableProcessors();
  private static final int MAX_QUERY_EXPRESSION_LOG_CHARS = 1000;
  private static final Logger LOG = Logger.getLogger(SkyQueryEnvironment.class.getName());

  private final BlazeTargetAccessor accessor = new BlazeTargetAccessor(this);
  protected final int loadingPhaseThreads;
  protected final WalkableGraphFactory graphFactory;
  protected final ImmutableList<String> universeScope;
  protected final String parserPrefix;
  protected final PathPackageLocator pkgPath;
  private final int queryEvaluationParallelismLevel;

  // The following fields are set in the #beforeEvaluateQuery method.
  private MultisetSemaphore<PackageIdentifier> packageSemaphore;
  protected WalkableGraph graph;
  private InterruptibleSupplier<ImmutableSet<PathFragment>> blacklistPatternsSupplier;
  private GraphBackedRecursivePackageProvider graphBackedRecursivePackageProvider;
  private ForkJoinPool forkJoinPool;
  private RecursivePackageProviderBackedTargetPatternResolver resolver;
  private final SkyKey universeKey;
  private final ImmutableList<TargetPatternKey> universeTargetPatternKeys;

  public SkyQueryEnvironment(
      boolean keepGoing,
      int loadingPhaseThreads,
      EventHandler eventHandler,
      Set<Setting> settings,
      Iterable<QueryFunction> extraFunctions,
      QueryExpressionEvalListener<Target> evalListener,
      String parserPrefix,
      WalkableGraphFactory graphFactory,
      List<String> universeScope,
      PathPackageLocator pkgPath) {
    this(
        keepGoing,
        loadingPhaseThreads,
        // SkyQueryEnvironment operates on a prepopulated Skyframe graph. Therefore, query
        // evaluation is completely CPU-bound.
        /*queryEvaluationParallelismLevel=*/ DEFAULT_THREAD_COUNT,
        eventHandler,
        settings,
        extraFunctions,
        evalListener,
        parserPrefix,
        graphFactory,
        universeScope,
        pkgPath);
  }

  protected SkyQueryEnvironment(
      boolean keepGoing,
      int loadingPhaseThreads,
      int queryEvaluationParallelismLevel,
      EventHandler eventHandler,
      Set<Setting> settings,
      Iterable<QueryFunction> extraFunctions,
      QueryExpressionEvalListener<Target> evalListener,
      String parserPrefix,
      WalkableGraphFactory graphFactory,
      List<String> universeScope,
      PathPackageLocator pkgPath) {
    super(
        keepGoing,
        /*strictScope=*/ true,
        /*labelFilter=*/ Rule.ALL_LABELS,
        eventHandler,
        settings,
        extraFunctions,
        evalListener);
    this.loadingPhaseThreads = loadingPhaseThreads;
    this.graphFactory = graphFactory;
    this.pkgPath = pkgPath;
    this.universeScope = ImmutableList.copyOf(Preconditions.checkNotNull(universeScope));
    this.parserPrefix = parserPrefix;
    Preconditions.checkState(
        !universeScope.isEmpty(), "No queries can be performed with an empty universe");
    this.queryEvaluationParallelismLevel = queryEvaluationParallelismLevel;
    this.universeKey = graphFactory.getUniverseKey(universeScope, parserPrefix);
    universeTargetPatternKeys =
        PrepareDepsOfPatternsFunction.getTargetPatternKeys(
            PrepareDepsOfPatternsFunction.getSkyKeys(universeKey, eventHandler));
  }

  private void beforeEvaluateQuery() throws InterruptedException {
    boolean resolverNeedsRecreation = false;
    if (graph == null || !graphFactory.isUpToDate(universeKey)) {
      // If this environment is uninitialized or the graph factory needs to evaluate, do so. We
      // assume here that this environment cannot be initialized-but-stale if the factory is up
      // to date.
      EvaluationResult<SkyValue> result;
      try (AutoProfiler p = AutoProfiler.logged("evaluation and walkable graph", LOG)) {
        result = graphFactory.prepareAndGet(universeKey, loadingPhaseThreads, eventHandler);
      }
      checkEvaluationResult(result);

      packageSemaphore = makeFreshPackageMultisetSemaphore();
      graph = result.getWalkableGraph();
      blacklistPatternsSupplier = InterruptibleSupplier.Memoize.of(new BlacklistSupplier(graph));

      graphBackedRecursivePackageProvider =
          new GraphBackedRecursivePackageProvider(graph, universeTargetPatternKeys, pkgPath);
      resolverNeedsRecreation = true;
    }
    if (forkJoinPool == null) {
      forkJoinPool =
          NamedForkJoinPool.newNamedPool("QueryEnvironment", queryEvaluationParallelismLevel);
      resolverNeedsRecreation = true;
    }
    if (resolverNeedsRecreation) {
      resolver =
          new RecursivePackageProviderBackedTargetPatternResolver(
              graphBackedRecursivePackageProvider,
              eventHandler,
              TargetPatternEvaluator.DEFAULT_FILTERING_POLICY,
              packageSemaphore);
    }
  }

  protected MultisetSemaphore<PackageIdentifier> makeFreshPackageMultisetSemaphore() {
    return MultisetSemaphore.unbounded();
  }

  @ThreadSafe
  public MultisetSemaphore<PackageIdentifier> getPackageMultisetSemaphore() {
    return packageSemaphore;
  }

  /**
   * The {@link EvaluationResult} is from the evaluation of a single PrepareDepsOfPatterns node. We
   * expect to see either a single successfully evaluated value or a cycle in the result.
   */
  private void checkEvaluationResult(EvaluationResult<SkyValue> result) {
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
      boolean foundCycle = !Iterables.isEmpty(result.getError().getCycleInfo());
      Preconditions.checkState(
          foundCycle,
          "Universe query \"%s\" failed with non-cycle error: %s",
          universeScope,
          result.getError());
    }
  }

  /**
   * A {@link QueryExpressionMapper} that transforms each occurrence of an expression of the form
   * {@literal 'rdeps(<universeScope>, <T>)'} to {@literal 'allrdeps(<T>)'}. The latter is more
   * efficient.
   */
  protected static class RdepsToAllRdepsQueryExpressionMapper extends QueryExpressionMapper {
    protected final TargetPattern.Parser targetPatternParser;
    private final String absoluteUniverseScopePattern;

    protected RdepsToAllRdepsQueryExpressionMapper(
        TargetPattern.Parser targetPatternParser,
        String universeScopePattern) {
      this.targetPatternParser = targetPatternParser;
      this.absoluteUniverseScopePattern = targetPatternParser.absolutize(universeScopePattern);
    }

    @Override
    public QueryExpression map(FunctionExpression functionExpression) {
      if (functionExpression.getFunction().getName().equals(new RdepsFunction().getName())) {
        List<Argument> args = functionExpression.getArgs();
        QueryExpression universeExpression = args.get(0).getExpression();
        if (universeExpression instanceof TargetLiteral) {
          TargetLiteral literalUniverseExpression = (TargetLiteral) universeExpression;
          String absolutizedUniverseExpression =
              targetPatternParser.absolutize(literalUniverseExpression.getPattern());
          if (absolutizedUniverseExpression.equals(absoluteUniverseScopePattern)) {
            List<Argument> argsTail = args.subList(1, functionExpression.getArgs().size());
            return new FunctionExpression(new AllRdepsFunction(), argsTail);
          }
        }
      }
      return super.map(functionExpression);
    }
  }

  @Override
  public final QueryExpression transformParsedQuery(QueryExpression queryExpression) {
    QueryExpressionMapper mapper = getQueryExpressionMapper();
    QueryExpression transformedQueryExpression = queryExpression.getMapped(mapper);
    LOG.info(String.format(
        "transformed query [%s] to [%s]",
        Ascii.truncate(
            queryExpression.toString(), MAX_QUERY_EXPRESSION_LOG_CHARS, "[truncated]"),
        Ascii.truncate(
            transformedQueryExpression.toString(), MAX_QUERY_EXPRESSION_LOG_CHARS, "[truncated]")));
    return transformedQueryExpression;
  }

  protected QueryExpressionMapper getQueryExpressionMapper() {
    if (universeScope.size() != 1) {
      return QueryExpressionMapper.identity();
    }
    TargetPattern.Parser targetPatternParser = new TargetPattern.Parser(parserPrefix);
    String universeScopePattern = Iterables.getOnlyElement(universeScope);
    return new RdepsToAllRdepsQueryExpressionMapper(targetPatternParser, universeScopePattern);
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
      if (throwableToThrow  != null) {
        LOG.log(Level.INFO, "About to shutdown FJP because of throwable", throwableToThrow);
        // Force termination of remaining tasks if evaluation failed abruptly (e.g. was
        // interrupted). We don't want to leave any dangling threads running tasks.
        forkJoinPool.shutdownNow();
      }
      forkJoinPool.awaitQuiescence(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
      if (throwableToThrow  != null) {
        // Signal that pool must be recreated on the next invocation.
        forkJoinPool = null;
        Throwables.propagateIfPossible(
            throwableToThrow, QueryException.class, InterruptedException.class);
      }
    }
  }

  @Override
  public QueryEvalResult evaluateQuery(
      QueryExpression expr, OutputFormatterCallback<Target> callback)
          throws QueryException, InterruptedException, IOException {
    // Some errors are reported as QueryExceptions and others as ERROR events (if --keep_going). The
    // result is set to have an error iff there were errors emitted during the query, so we reset
    // errors here.
    eventHandler.resetErrors();
    beforeEvaluateQuery();

    // SkyQueryEnvironment batches callback invocations using a BatchStreamedCallback, created here
    // so that there's one per top-level evaluateQuery call. The batch size is large enough that
    // per-call costs of calling the original callback are amortized over a good number of targets,
    // and small enough that holding a batch of targets in memory doesn't risk an OOM error.
    //
    // This flushes the batched callback prior to constructing the QueryEvalResult in the unlikely
    // case of a race between the original callback and the eventHandler.
    BatchStreamedCallback batchCallback = new BatchStreamedCallback(callback, BATCH_CALLBACK_SIZE);
    return super.evaluateQuery(expr, batchCallback);
  }

  private Map<SkyKey, Collection<Target>> targetifyValues(
      Map<SkyKey, ? extends Iterable<SkyKey>> input) throws InterruptedException {
    return targetifyValues(
        input,
        makePackageKeyToTargetKeyMap(ImmutableSet.copyOf(Iterables.concat(input.values()))));
  }

  private Map<SkyKey, Collection<Target>> targetifyValues(
      Map<SkyKey, ? extends Iterable<SkyKey>> input,
      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap) throws InterruptedException {
    ImmutableMap.Builder<SkyKey, Collection<Target>> result = ImmutableMap.builder();

    Map<SkyKey, Target> allTargets =
        makeTargetsFromPackageKeyToTargetKeyMap(packageKeyToTargetKeyMap);

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

  private Map<Target, Collection<Target>> targetifyKeys(Map<SkyKey, Collection<Target>> input)
      throws InterruptedException {
    Map<SkyKey, Target> targets = makeTargetsFromSkyKeys(input.keySet());
    ImmutableMap.Builder<Target, Collection<Target>> resultBuilder = ImmutableMap.builder();
    for (Map.Entry<SkyKey, Collection<Target>> entry : input.entrySet()) {
      SkyKey key = entry.getKey();
      Target target = targets.get(key);
      if (target != null) {
        resultBuilder.put(target, entry.getValue());
      }
    }
    return resultBuilder.build();
  }

  private Map<Target, Collection<Target>> targetifyKeysAndValues(
      Map<SkyKey, Iterable<SkyKey>> input) throws InterruptedException {
    return targetifyKeys(targetifyValues(input));
  }

  private Map<Target, Collection<Target>> getRawFwdDeps(Iterable<Target> targets)
      throws InterruptedException {
    return targetifyKeysAndValues(graph.getDirectDeps(makeTransitiveTraversalKeys(targets)));
  }

  private Map<SkyKey, Collection<Target>> getRawReverseDeps(
      Iterable<SkyKey> transitiveTraversalKeys) throws InterruptedException {
    return targetifyValues(graph.getReverseDeps(transitiveTraversalKeys));
  }

  private Set<Label> getAllowedDeps(Rule rule) throws InterruptedException {
    Set<Label> allowedLabels = new HashSet<>(rule.getTransitions(dependencyFilter).values());
    allowedLabels.addAll(rule.getVisibility().getDependencyLabels());
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

  /** Targets may not be in the graph because they are not in the universe or depend on cycles. */
  private void warnIfMissingTargets(
      Iterable<Target> targets, Set<Target> result) {
    if (Iterables.size(targets) != result.size()) {
      Set<Target> missingTargets = Sets.difference(ImmutableSet.copyOf(targets), result);
      eventHandler.handle(Event.warn("Targets were missing from graph: " + missingTargets));
    }
  }

  @Override
  public Collection<Target> getFwdDeps(Iterable<Target> targets) throws InterruptedException {
    Set<Target> result = new HashSet<>();
    Map<Target, Collection<Target>> rawFwdDeps = getRawFwdDeps(targets);
    warnIfMissingTargets(targets, rawFwdDeps.keySet());
    for (Map.Entry<Target, Collection<Target>> entry : rawFwdDeps.entrySet()) {
      result.addAll(filterFwdDeps(entry.getKey(), entry.getValue()));
    }
    return result;
  }

  @Override
  public Collection<Target> getReverseDeps(Iterable<Target> targets) throws InterruptedException {
    return getReverseDepsOfTransitiveTraversalKeys(Iterables.transform(targets, TARGET_TO_SKY_KEY));
  }

  Collection<Target> getReverseDepsOfTransitiveTraversalKeys(
      Iterable<SkyKey> transitiveTraversalKeys) throws InterruptedException {
    Map<SkyKey, Collection<Target>> rawReverseDeps = getRawReverseDeps(transitiveTraversalKeys);
    return processRawReverseDeps(rawReverseDeps);
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
  public Set<Target> getTransitiveClosure(Set<Target> targets) throws InterruptedException {
    Set<Target> visited = new HashSet<>();
    Collection<Target> current = targets;
    while (!current.isEmpty()) {
      Collection<Target> toVisit = Collections2.filter(current,
          Predicates.not(Predicates.in(visited)));
      current = getFwdDeps(toVisit);
      visited.addAll(toVisit);
    }
    return ImmutableSet.copyOf(visited);
  }

  // Implemented with a breadth-first search.
  @Override
  public Set<Target> getNodesOnPath(Target from, Target to) throws InterruptedException {
    // Tree of nodes visited so far.
    Map<Target, Target> nodeToParent = new HashMap<>();
    // Contains all nodes left to visit in a (LIFO) stack.
    Deque<Target> toVisit = new ArrayDeque<>();
    toVisit.add(from);
    nodeToParent.put(from, null);
    while (!toVisit.isEmpty()) {
      Target current = toVisit.removeFirst();
      if (to.equals(current)) {
        return ImmutableSet.copyOf(Digraph.getPathToTreeNode(nodeToParent, to));
      }
      for (Target dep : getFwdDeps(ImmutableList.of(current))) {
        if (!nodeToParent.containsKey(dep)) {
          nodeToParent.put(dep, current);
          toVisit.addFirst(dep);
        }
      }
    }
    // Note that the only current caller of this method checks first to see if there is a path
    // before calling this method. It is not clear what the return value should be here.
    return null;
  }

  @ThreadSafe
  @Override
  public void eval(QueryExpression expr, VariableContext<Target> context, Callback<Target> callback)
      throws QueryException, InterruptedException {
    // TODO(bazel-team): Refactor QueryEnvironment et al. such that this optimization is enabled for
    // all QueryEnvironment implementations.
    if (callback instanceof ThreadSafeCallback) {
      expr.parEval(this, context, (ThreadSafeCallback<Target>) callback, forkJoinPool);
    } else {
      expr.eval(this, context, callback);
    }
  }

  @ThreadSafe
  @Override
  public ThreadSafeUniquifier<Target> createUniquifier() {
    return createTargetUniquifier();
  }

  @ThreadSafe
  ThreadSafeUniquifier<Target> createTargetUniquifier() {
    return new ThreadSafeTargetUniquifier(DEFAULT_THREAD_COUNT);
  }

  @ThreadSafe
  ThreadSafeUniquifier<SkyKey> createSkyKeyUniquifier() {
    return new ThreadSafeSkyKeyUniquifier(DEFAULT_THREAD_COUNT);
  }

  @ThreadSafe
  ThreadSafeUniquifier<Pair<SkyKey, SkyKey>> createReverseDepSkyKeyUniquifier() {
    return new ThreadSafeReverseDepSkyKeyUniquifier(DEFAULT_THREAD_COUNT);
  }

  private Pair<TargetPattern, ImmutableSet<PathFragment>> getPatternAndExcludes(String pattern)
      throws TargetParsingException, InterruptedException {
    TargetPatternKey targetPatternKey =
        ((TargetPatternKey)
            TargetPatternValue.key(
                    pattern, TargetPatternEvaluator.DEFAULT_FILTERING_POLICY, parserPrefix)
                .argument());
    ImmutableSet<PathFragment> subdirectoriesToExclude =
        targetPatternKey.getAllSubdirectoriesToExclude(blacklistPatternsSupplier);
    return Pair.of(targetPatternKey.getParsedPattern(), subdirectoriesToExclude);
  }

  @ThreadSafe
  @Override
  public void getTargetsMatchingPattern(
      QueryExpression owner, String pattern, Callback<Target> callback)
      throws QueryException, InterruptedException {
    // Directly evaluate the target pattern, making use of packages in the graph.
    try {
      Pair<TargetPattern, ImmutableSet<PathFragment>> patternToEvalAndSubdirectoriesToExclude =
          getPatternAndExcludes(pattern);
      TargetPattern patternToEval = patternToEvalAndSubdirectoriesToExclude.getFirst();
      ImmutableSet<PathFragment> subdirectoriesToExclude =
          patternToEvalAndSubdirectoriesToExclude.getSecond();
      patternToEval.eval(resolver, subdirectoriesToExclude, callback, QueryException.class);
    } catch (TargetParsingException e) {
      reportBuildFileError(owner, e.getMessage());
    }
  }

  @Override
  public void getTargetsMatchingPatternPar(
      QueryExpression owner,
      String pattern,
      ThreadSafeCallback<Target> callback,
      ForkJoinPool forkJoinPool)
      throws QueryException, InterruptedException {
    // Directly evaluate the target pattern, making use of packages in the graph.
    try {
      Pair<TargetPattern, ImmutableSet<PathFragment>> patternToEvalAndSubdirectoriesToExclude =
          getPatternAndExcludes(pattern);
      TargetPattern patternToEval = patternToEvalAndSubdirectoriesToExclude.getFirst();
      ImmutableSet<PathFragment> subdirectoriesToExclude =
          patternToEvalAndSubdirectoriesToExclude.getSecond();
      patternToEval.parEval(
          resolver, subdirectoriesToExclude, callback, QueryException.class, forkJoinPool);
    } catch (TargetParsingException e) {
      reportBuildFileError(owner, e.getMessage());
    }
  }

  @ThreadSafe
  @Override
  public Set<Target> getBuildFiles(
      QueryExpression caller,
      Set<Target> nodes,
      boolean buildFiles,
      boolean subincludes,
      boolean loads)
      throws QueryException {
    Set<Target> dependentFiles = new LinkedHashSet<>();
    Set<Package> seenPackages = new HashSet<>();
    // Keep track of seen labels, to avoid adding a fake subinclude label that also exists as a
    // real target.
    Set<Label> seenLabels = new HashSet<>();

    // Adds all the package definition files (BUILD files and build
    // extensions) for package "pkg", to "buildfiles".
    for (Target x : nodes) {
      Package pkg = x.getPackage();
      if (seenPackages.add(pkg)) {
        if (buildFiles) {
          addIfUniqueLabel(pkg.getBuildFile(), seenLabels, dependentFiles);
        }

        List<Label> extensions = new ArrayList<>();
        if (subincludes) {
          extensions.addAll(pkg.getSubincludeLabels());
        }
        if (loads) {
          extensions.addAll(pkg.getSkylarkFileDependencies());
        }

        for (Label subinclude : extensions) {
          addIfUniqueLabel(getSubincludeTarget(subinclude, pkg), seenLabels, dependentFiles);

          if (buildFiles) {
            // Also add the BUILD file of the subinclude.
            try {
              addIfUniqueLabel(
                  getSubincludeTarget(subinclude.getLocalTargetLabel("BUILD"), pkg),
                  seenLabels,
                  dependentFiles);

            } catch (LabelSyntaxException e) {
              throw new AssertionError("BUILD should always parse as a target name", e);
            }
          }
        }
      }
    }
    return dependentFiles;
  }

  private static void addIfUniqueLabel(Target node, Set<Label> labels, Set<Target> nodes) {
    if (labels.add(node.getLabel())) {
      nodes.add(node);
    }
  }

  private static Target getSubincludeTarget(Label label, Package pkg) {
    return new FakeSubincludeTarget(label, pkg);
  }

  @ThreadSafe
  @Override
  public TargetAccessor<Target> getAccessor() {
    return accessor;
  }

  @ThreadSafe
  @Override
  public Target getTarget(Label label)
      throws TargetNotFoundException, QueryException, InterruptedException {
    SkyKey packageKey = PackageValue.key(label.getPackageIdentifier());
    if (!graph.exists(packageKey)) {
      throw new QueryException(packageKey + " does not exist in graph");
    }
    try {
      PackageValue packageValue = (PackageValue) graph.getValue(packageKey);
      if (packageValue != null) {
        Package pkg = packageValue.getPackage();
        if (pkg.containsErrors()) {
          throw new BuildFileContainsErrorsException(label.getPackageIdentifier());
        }
        return packageValue.getPackage().getTarget(label.getName());
      } else {
        throw (NoSuchThingException) Preconditions.checkNotNull(
            graph.getException(packageKey), label);
      }
    } catch (NoSuchThingException e) {
      throw new TargetNotFoundException(e);
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
  public void buildTransitiveClosure(QueryExpression caller, Set<Target> targets, int maxDepth)
      throws QueryException, InterruptedException {
    // Everything has already been loaded, so here we just check for errors so that we can
    // pre-emptively throw/report if needed.
    Iterable<SkyKey> transitiveTraversalKeys = makeTransitiveTraversalKeys(targets);
    ImmutableList.Builder<String> errorMessagesBuilder = ImmutableList.builder();

    // First, look for errors in the successfully evaluated TransitiveTraversalValues. They may
    // have encountered errors that they were able to recover from.
    Set<Entry<SkyKey, SkyValue>> successfulEntries =
        graph.getSuccessfulValues(transitiveTraversalKeys).entrySet();
    Builder<SkyKey> successfulKeysBuilder = ImmutableSet.builder();
    for (Entry<SkyKey, SkyValue> successfulEntry : successfulEntries) {
      successfulKeysBuilder.add(successfulEntry.getKey());
      TransitiveTraversalValue value = (TransitiveTraversalValue) successfulEntry.getValue();
      String firstErrorMessage = value.getFirstErrorMessage();
      if (firstErrorMessage != null) {
        errorMessagesBuilder.add(firstErrorMessage);
      }
    }
    ImmutableSet<SkyKey> successfulKeys = successfulKeysBuilder.build();

    // Next, look for errors from the unsuccessfully evaluated TransitiveTraversal skyfunctions.
    Iterable<SkyKey> unsuccessfulKeys =
        Iterables.filter(transitiveTraversalKeys, Predicates.not(Predicates.in(successfulKeys)));
    Set<Entry<SkyKey, Exception>> errorEntries =
        graph.getMissingAndExceptions(unsuccessfulKeys).entrySet();
    for (Map.Entry<SkyKey, Exception> entry : errorEntries) {
      if (entry.getValue() == null) {
        // Targets may be in the graph because they are not in the universe or depend on cycles.
        eventHandler.handle(Event.warn(entry.getKey().argument() + " does not exist in graph"));
      } else {
        errorMessagesBuilder.add(entry.getValue().getMessage());
      }
    }

    // Lastly, report all found errors.
    ImmutableList<String> errorMessages = errorMessagesBuilder.build();
    for (String errorMessage : errorMessages) {
      reportBuildFileError(caller, errorMessage);
    }
  }

  @Override
  protected void preloadOrThrow(QueryExpression caller, Collection<String> patterns)
      throws QueryException, TargetParsingException {
    // SkyQueryEnvironment directly evaluates target patterns in #getTarget and similar methods
    // using its graph, which is prepopulated using the universeScope (see #beforeEvaluateQuery),
    // so no preloading of target patterns is necessary.
  }

  static final Function<SkyKey, Label> SKYKEY_TO_LABEL =
      new Function<SkyKey, Label>() {
        @Nullable
        @Override
        public Label apply(SkyKey skyKey) {
          SkyFunctionName functionName = skyKey.functionName();
          if (!functionName.equals(SkyFunctions.TRANSITIVE_TRAVERSAL)) {
            // Skip non-targets.
            return null;
          }
          return (Label) skyKey.argument();
        }
      };

  static final Function<SkyKey, PackageIdentifier> PACKAGE_SKYKEY_TO_PACKAGE_IDENTIFIER =
      new Function<SkyKey, PackageIdentifier>() {
        @Override
        public PackageIdentifier apply(SkyKey skyKey) {
          return (PackageIdentifier) skyKey.argument();
        }
      };

  @ThreadSafe
  Multimap<SkyKey, SkyKey> makePackageKeyToTargetKeyMap(Iterable<SkyKey> keys) {
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

  @ThreadSafe
  public Map<SkyKey, Target> makeTargetsFromSkyKeys(Iterable<SkyKey> keys)
      throws InterruptedException {
    return makeTargetsFromPackageKeyToTargetKeyMap(makePackageKeyToTargetKeyMap(keys));
  }

  @ThreadSafe
  public Map<SkyKey, Target> makeTargetsFromPackageKeyToTargetKeyMap(
      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap) throws InterruptedException {
    ImmutableMap.Builder<SkyKey, Target> result = ImmutableMap.builder();
    Set<SkyKey> processedTargets = new HashSet<>();
    Map<SkyKey, SkyValue> packageMap = graph.getSuccessfulValues(packageKeyToTargetKeyMap.keySet());
    for (Map.Entry<SkyKey, SkyValue> entry : packageMap.entrySet()) {
      for (SkyKey targetKey : packageKeyToTargetKeyMap.get(entry.getKey())) {
        if (processedTargets.add(targetKey)) {
          try {
            result.put(
                targetKey,
                ((PackageValue) entry.getValue())
                    .getPackage()
                    .getTarget((SKYKEY_TO_LABEL.apply(targetKey)).getName()));
          } catch (NoSuchTargetException e) {
            // Skip missing target.
          }
        }
      }
    }
    return result.build();
  }

  static final Function<Target, SkyKey> TARGET_TO_SKY_KEY =
      new Function<Target, SkyKey>() {
        @Override
        public SkyKey apply(Target target) {
          return TransitiveTraversalValue.key(target.getLabel());
        }
      };

  /** A strict (i.e. non-lazy) variant of {@link #makeTransitiveTraversalKeys}. */
  public static Iterable<SkyKey> makeTransitiveTraversalKeysStrict(Iterable<Target> targets) {
    return ImmutableList.copyOf(makeTransitiveTraversalKeys(targets));
  }

  private static Iterable<SkyKey> makeTransitiveTraversalKeys(Iterable<Target> targets) {
    return Iterables.transform(targets, TARGET_TO_SKY_KEY);
  }

  @Override
  public Target getOrCreate(Target target) {
    return target;
  }

  /**
   * Returns package lookup keys for looking up the package root for which there may be a relevant
   * (from the perspective of {@link #getRBuildFiles}) {@link FileValue} node in the graph for
   * {@code originalFileFragment}, which is assumed to be a file path.
   *
   * <p>This is a helper function for {@link #getSkyKeysForFileFragments}.
   */
  private static Iterable<SkyKey> getPkgLookupKeysForFile(PathFragment originalFileFragment,
      PathFragment currentPathFragment) {
    if (originalFileFragment.equals(currentPathFragment)
        && originalFileFragment.equals(Label.EXTERNAL_PACKAGE_FILE_NAME)) {
      Preconditions.checkState(
          Label.EXTERNAL_PACKAGE_FILE_NAME.getParentDirectory().equals(PathFragment.EMPTY_FRAGMENT),
          Label.EXTERNAL_PACKAGE_FILE_NAME);
      return ImmutableList.of(
          PackageLookupValue.key(Label.EXTERNAL_PACKAGE_IDENTIFIER),
          PackageLookupValue.key(PackageIdentifier.createInMainRepo(PathFragment.EMPTY_FRAGMENT)));
    }
    PathFragment parentPathFragment = currentPathFragment.getParentDirectory();
    return parentPathFragment == null
        ? ImmutableList.<SkyKey>of()
        : ImmutableList.of(PackageLookupValue.key(
            PackageIdentifier.createInMainRepo(parentPathFragment)));
  }

  /**
   * Returns FileValue keys for which there may be relevant (from the perspective of {@link
   * #getRBuildFiles}) FileValues in the graph corresponding to the given {@code pathFragments},
   * which are assumed to be file paths.
   *
   * <p>To do this, we emulate the {@link ContainingPackageLookupFunction} logic: for each given
   * file path, we look for the nearest ancestor directory (starting with its parent directory), if
   * any, that has a package. The {@link PackageLookupValue} for this package tells us the package
   * root that we should use for the {@link RootedPath} for the {@link FileValue} key.
   *
   * <p>Note that there may not be nodes in the graph corresponding to the returned SkyKeys.
   */
  Collection<SkyKey> getSkyKeysForFileFragments(Iterable<PathFragment> pathFragments)
      throws InterruptedException {
    Set<SkyKey> result = new HashSet<>();
    Multimap<PathFragment, PathFragment> currentToOriginal = ArrayListMultimap.create();
    for (PathFragment pathFragment : pathFragments) {
      currentToOriginal.put(pathFragment, pathFragment);
    }
    while (!currentToOriginal.isEmpty()) {
      Multimap<SkyKey, PathFragment> packageLookupKeysToOriginal = ArrayListMultimap.create();
      Multimap<SkyKey, PathFragment> packageLookupKeysToCurrent = ArrayListMultimap.create();
      for (Entry<PathFragment, PathFragment> entry : currentToOriginal.entries()) {
        PathFragment current = entry.getKey();
        PathFragment original = entry.getValue();
        for (SkyKey packageLookupKey : getPkgLookupKeysForFile(original, current)) {
          packageLookupKeysToOriginal.put(packageLookupKey, original);
          packageLookupKeysToCurrent.put(packageLookupKey, current);
        }
      }
      Map<SkyKey, SkyValue> lookupValues =
          graph.getSuccessfulValues(packageLookupKeysToOriginal.keySet());
      for (Map.Entry<SkyKey, SkyValue> entry : lookupValues.entrySet()) {
        SkyKey packageLookupKey = entry.getKey();
        PackageLookupValue packageLookupValue = (PackageLookupValue) entry.getValue();
        if (packageLookupValue.packageExists()) {
          Collection<PathFragment> originalFiles =
              packageLookupKeysToOriginal.get(packageLookupKey);
          Preconditions.checkState(!originalFiles.isEmpty(), entry);
          for (PathFragment fileName : originalFiles) {
            result.add(
                FileValue.key(RootedPath.toRootedPath(packageLookupValue.getRoot(), fileName)));
          }
          for (PathFragment current : packageLookupKeysToCurrent.get(packageLookupKey)) {
            currentToOriginal.removeAll(current);
          }
        }
      }
      Multimap<PathFragment, PathFragment> newCurrentToOriginal = ArrayListMultimap.create();
      for (PathFragment pathFragment : currentToOriginal.keySet()) {
        PathFragment parent = pathFragment.getParentDirectory();
        if (parent != null) {
          newCurrentToOriginal.putAll(parent, currentToOriginal.get(pathFragment));
        }
      }
      currentToOriginal = newCurrentToOriginal;
    }
    return result;
  }

  private static final Function<SkyValue, Package> EXTRACT_PACKAGE =
      new Function<SkyValue, Package>() {
        @Override
        public Package apply(SkyValue skyValue) {
          return ((PackageValue) skyValue).getPackage();
        }
      };

  private static final Predicate<Package> ERROR_FREE_PACKAGE =
      new Predicate<Package>() {
        @Override
        public boolean apply(Package pkg) {
          return !pkg.containsErrors();
        }
      };

  private static final Function<Package, Target> GET_BUILD_FILE =
      new Function<Package, Target>() {
        @Override
        public Target apply(Package pkg) {
          return pkg.getBuildFile();
        }
      };

  static Iterable<Target> getBuildFilesForPackageValues(Iterable<SkyValue> packageValues) {
    return Iterables.transform(
        Iterables.filter(Iterables.transform(packageValues, EXTRACT_PACKAGE), ERROR_FREE_PACKAGE),
        GET_BUILD_FILE);
  }

  @ThreadSafe
  void getRBuildFilesParallel(
      Collection<PathFragment> fileIdentifiers,
      ThreadSafeCallback<Target> callback,
      ForkJoinPool forkJoinPool)
      throws QueryException, InterruptedException {
    ParallelSkyQueryUtils.getRBuildFilesParallel(this, fileIdentifiers, callback, packageSemaphore);
  }

  /**
   * Calculates the set of {@link Package} objects, represented as source file targets, that depend
   * on the given list of BUILD files and subincludes (other files are filtered out).
   */
  @ThreadSafe
  void getRBuildFiles(Collection<PathFragment> fileIdentifiers, Callback<Target> callback)
      throws QueryException, InterruptedException {
    Collection<SkyKey> files = getSkyKeysForFileFragments(fileIdentifiers);
    Uniquifier<SkyKey> keyUniquifier = new ThreadSafeSkyKeyUniquifier(/*concurrencyLevel=*/ 1);
    Collection<SkyKey> current = keyUniquifier.unique(graph.getSuccessfulValues(files).keySet());
    Set<SkyKey> resultKeys = CompactHashSet.create();
    while (!current.isEmpty()) {
      Collection<Iterable<SkyKey>> reverseDeps = graph.getReverseDeps(current).values();
      current = new HashSet<>();
      for (SkyKey rdep : Iterables.concat(reverseDeps)) {
        if (rdep.functionName().equals(SkyFunctions.PACKAGE)) {
          resultKeys.add(rdep);
          // Every package has a dep on the external package, so we need to include those edges too.
          if (rdep.equals(PackageValue.key(Label.EXTERNAL_PACKAGE_IDENTIFIER))) {
            if (keyUniquifier.unique(rdep)) {
              current.add(rdep);
            }
          }
        } else if (!rdep.functionName().equals(SkyFunctions.PACKAGE_LOOKUP)) {
          // Packages may depend on the existence of subpackages, but these edges aren't relevant to
          // rbuildfiles.
          if (keyUniquifier.unique(rdep)) {
            current.add(rdep);
          }
        }
      }
      if (resultKeys.size() >= BATCH_CALLBACK_SIZE) {
        for (Iterable<SkyKey> batch : Iterables.partition(resultKeys, BATCH_CALLBACK_SIZE)) {
          callback.process(
              getBuildFilesForPackageValues(graph.getSuccessfulValues(batch).values()));
        }
        resultKeys.clear();
      }
    }
    callback.process(getBuildFilesForPackageValues(graph.getSuccessfulValues(resultKeys).values()));
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

    BlacklistSupplier(WalkableGraph graph) {
      this.graph = graph;
    }

    @Override
    public ImmutableSet<PathFragment> get() throws InterruptedException {
      return ((BlacklistedPackagePrefixesValue)
              graph.getValue(BlacklistedPackagePrefixesValue.key()))
          .getPatterns();
    }
  }

  private static class ThreadSafeTargetUniquifier
      extends AbstractThreadSafeUniquifier<Target, Label> {
    protected ThreadSafeTargetUniquifier(int concurrencyLevel) {
      super(concurrencyLevel);
    }

    @Override
    protected Label extractKey(Target element) {
      return element.getLabel();
    }
  }

  private static class ThreadSafeSkyKeyUniquifier
      extends AbstractThreadSafeUniquifier<SkyKey, SkyKey> {
    protected ThreadSafeSkyKeyUniquifier(int concurrencyLevel) {
      super(concurrencyLevel);
    }

    @Override
    protected SkyKey extractKey(SkyKey element) {
      return element;
    }
  }

  /**
   * A uniquifer which takes a pair of parent and reverse dep, and uniquify based on the second
   * element (reverse dep).
   */
  private static class ThreadSafeReverseDepSkyKeyUniquifier
      extends AbstractThreadSafeUniquifier<Pair<SkyKey, SkyKey>, SkyKey> {
    protected ThreadSafeReverseDepSkyKeyUniquifier(int concurrencyLevel) {
      super(concurrencyLevel);
    }

    @Override
    protected SkyKey extractKey(Pair<SkyKey, SkyKey> element) {
      return element.second;
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
  @ThreadSafe
  private static class BatchStreamedCallback extends OutputFormatterCallback<Target>
      implements ThreadSafeCallback<Target> {

    private final OutputFormatterCallback<Target> callback;
    private final ThreadSafeUniquifier<Target> uniquifier =
        new ThreadSafeTargetUniquifier(DEFAULT_THREAD_COUNT);
    private final Object pendingLock = new Object();
    private List<Target> pending = new ArrayList<>();
    private int batchThreshold;

    private BatchStreamedCallback(OutputFormatterCallback<Target> callback, int batchThreshold) {
      this.callback = callback;
      this.batchThreshold = batchThreshold;
    }

    @Override
    public void start() throws IOException {
      callback.start();
    }

    @Override
    public void processOutput(Iterable<Target> partialResult)
        throws IOException, InterruptedException {
      ImmutableList<Target> uniquifiedTargets = uniquifier.unique(partialResult);
      synchronized (pendingLock) {
        Preconditions.checkNotNull(pending, "Reuse of the callback is not allowed");
        pending.addAll(uniquifiedTargets);
        if (pending.size() >= batchThreshold) {
          callback.processOutput(pending);
          pending = new ArrayList<>();
        }
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
  public void getAllRdepsUnboundedParallel(
      QueryExpression expression,
      VariableContext<Target> context,
      ThreadSafeCallback<Target> callback,
      ForkJoinPool forkJoinPool)
      throws QueryException, InterruptedException {
    ParallelSkyQueryUtils.getAllRdepsUnboundedParallel(
        this, expression, context, callback, packageSemaphore);
  }

  @ThreadSafe
  @Override
  public void getAllRdeps(
      QueryExpression expression,
      Predicate<Target> universe,
      VariableContext<Target> context,
      Callback<Target> callback,
      int depth)
      throws QueryException, InterruptedException {
    getAllRdeps(expression, universe, context, callback, depth, BATCH_CALLBACK_SIZE);
  }

  /**
   * Computes and applies the callback to the reverse dependencies of the expression.
   *
   * <p>Batch size is used to only populate at most N targets at one time, because some individual
   * nodes are directly depended on by a large number of other nodes.
   */
  @VisibleForTesting
  protected void getAllRdeps(
      QueryExpression expression,
      Predicate<Target> universe,
      VariableContext<Target> context,
      Callback<Target> callback,
      int depth,
      int batchSize)
      throws QueryException, InterruptedException {
    Uniquifier<Target> uniquifier = createUniquifier();
    eval(
        expression,
        context,
        new BatchAllRdepsCallback(uniquifier, universe, callback, depth, batchSize));
  }

  private class BatchAllRdepsCallback implements Callback<Target> {
    private final Uniquifier<Target> uniquifier;
    private final Predicate<Target> universe;
    private final Callback<Target> callback;
    private final int depth;
    private final int batchSize;

    private BatchAllRdepsCallback(
        Uniquifier<Target> uniquifier,
        Predicate<Target> universe,
        Callback<Target> callback,
        int depth,
        int batchSize) {
      this.uniquifier = uniquifier;
      this.universe = universe;
      this.callback = callback;
      this.depth = depth;
      this.batchSize = batchSize;
    }

    @Override
    public void process(Iterable<Target> targets) throws QueryException, InterruptedException {
      Iterable<Target> currentInUniverse = Iterables.filter(targets, universe);
      ImmutableList<Target> uniqueTargets = uniquifier.unique(currentInUniverse);
      callback.process(uniqueTargets);

      // Maintain a queue to allow tracking rdep relationships in BFS order. Rdeps are stored
      // as 1:N SkyKey mappings instead of fully populated Targets to save memory. Targets
      // have a reference to their entire Package, which is really memory expensive.
      Queue<Map.Entry<SkyKey, Iterable<SkyKey>>> reverseDepsQueue = new LinkedList<>();
      reverseDepsQueue.addAll(
          graph.getReverseDeps(makeTransitiveTraversalKeys(uniqueTargets)).entrySet());

      // In each iteration, we populate a size-limited (no more than batchSize) number of
      // SkyKey mappings to targets, and append the SkyKey rdeps mappings to the queue. Once
      // processed by callback, the targets are dequeued and not referenced any more, making
      // them available for garbage collection.

      for (int i = 0; i < depth; i++) {
        // The mappings between nodes and their reverse deps must be preserved instead of the
        // reverse deps alone. Later when deserializing dependent nodes using SkyKeys, we need to
        // check if their allowed deps contain the dependencies.
        Map<SkyKey, Iterable<SkyKey>> reverseDepsMap = Maps.newHashMap();
        int batch = 0; // Tracking the current total number of rdeps in reverseDepsMap.
        int processed = 0;
        // Save current size as when we are process nodes in the current level, new mappings (for
        // the next level) are added to the queue.
        int size = reverseDepsQueue.size();
        while (processed < size) {
          // We always peek the first element in the queue without polling it, to determine if
          // adding it to the pending list will break the limit of max size. If yes then we process
          // and empty the pending list first, and poll the element in the next iteration.
          Map.Entry<SkyKey, Iterable<SkyKey>> entry = reverseDepsQueue.peek();

          // The value of the entry is either a CompactHashSet or ImmutableList, which can return
          // the size in O(1) time.
          int rdepsSize = Iterables.size(entry.getValue());
          if (rdepsSize == 0) {
            reverseDepsQueue.poll();
            processed++;
            continue;
          }

          if ((rdepsSize + batch <= batchSize)) {
            // If current size is less than batch size, dequeue the node, update the current
            // batch size and map.
            reverseDepsMap.put(entry.getKey(), entry.getValue());
            batch += rdepsSize;
            reverseDepsQueue.poll();
            processed++;
          } else {
            if (batch == 0) {
              // The (single) node has more rdeps than the limit, divide them up to process
              // separately.
              for (Iterable<SkyKey> subList : Iterables.partition(entry.getValue(), batchSize)) {
                reverseDepsMap.put(entry.getKey(), subList);
                processReverseDepsMap(uniquifier, reverseDepsMap, callback, reverseDepsQueue);
              }

              reverseDepsQueue.poll();
              processed++;
            } else {
              // There are some nodes in the pending process list. Process them first and come
              // back to this node later (in next iteration).
              processReverseDepsMap(uniquifier, reverseDepsMap, callback, reverseDepsQueue);
              batch = 0;
            }
          }
        }

        if (!reverseDepsMap.isEmpty()) {
          processReverseDepsMap(uniquifier, reverseDepsMap, callback, reverseDepsQueue);
        }

        // If the queue is empty after all nodes in the current level are processed, stop
        // processing as there are no more reverse deps.
        if (reverseDepsQueue.isEmpty()) {
          break;
        }
      }
    }

    /**
     * Populates {@link Target}s from reverse dep mappings of {@link SkyKey}s, empties the pending
     * list and add next level reverse dep mappings of {@link SkyKey}s to the queue.
     */
    private void processReverseDepsMap(
        Uniquifier<Target> uniquifier,
        Map<SkyKey, Iterable<SkyKey>> reverseDepsMap,
        Callback<Target> callback,
        Queue<Map.Entry<SkyKey, Iterable<SkyKey>>> reverseDepsQueue)
        throws QueryException, InterruptedException {
      Collection<Target> children = processRawReverseDeps(targetifyValues(reverseDepsMap));
      Iterable<Target> currentInUniverse = Iterables.filter(children, universe);
      ImmutableList<Target> uniqueChildren = uniquifier.unique(currentInUniverse);
      reverseDepsMap.clear();

      if (!uniqueChildren.isEmpty()) {
        callback.process(uniqueChildren);
        reverseDepsQueue.addAll(
            graph.getReverseDeps(makeTransitiveTraversalKeys(uniqueChildren)).entrySet());
      }
    }
  }
}
