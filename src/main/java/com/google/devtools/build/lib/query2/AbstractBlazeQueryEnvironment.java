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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.ErrorSensingEventHandler;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.pkgcache.TransitivePackageLoader;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AggregateAllCallback;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.WalkableGraph.WalkableGraphFactory;

import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * {@link QueryEnvironment} that can evaluate queries to produce a result, and implements as much
 * of QueryEnvironment as possible while remaining mostly agnostic as to the objects being stored.
 */
public abstract class AbstractBlazeQueryEnvironment<T> implements QueryEnvironment<T> {
  protected final ErrorSensingEventHandler eventHandler;
  private final Map<String, Set<T>> letBindings = new HashMap<>();
  protected final boolean keepGoing;
  protected final boolean strictScope;

  protected final DependencyFilter dependencyFilter;
  private final Predicate<Label> labelFilter;

  private final Set<Setting> settings;
  private final List<QueryFunction> extraFunctions;

 private static final Logger LOG = Logger.getLogger(AbstractBlazeQueryEnvironment.class.getName());

  protected AbstractBlazeQueryEnvironment(boolean keepGoing,
      boolean strictScope,
      Predicate<Label> labelFilter,
      EventHandler eventHandler,
      Set<Setting> settings,
      Iterable<QueryFunction> extraFunctions) {
    this.eventHandler = new ErrorSensingEventHandler(eventHandler);
    this.keepGoing = keepGoing;
    this.strictScope = strictScope;
    this.dependencyFilter = constructDependencyFilter(settings);
    this.labelFilter = labelFilter;
    this.settings = Sets.immutableEnumSet(settings);
    this.extraFunctions = ImmutableList.copyOf(extraFunctions);
  }

  private static DependencyFilter constructDependencyFilter(Set<Setting> settings) {
    DependencyFilter specifiedFilter =
        settings.contains(Setting.NO_HOST_DEPS)
            ? DependencyFilter.NO_HOST_DEPS
            : DependencyFilter.ALL_DEPS;
    if (settings.contains(Setting.NO_IMPLICIT_DEPS)) {
      specifiedFilter = DependencyFilter.and(specifiedFilter, DependencyFilter.NO_IMPLICIT_DEPS);
    }
    if (settings.contains(Setting.NO_NODEP_DEPS)) {
      specifiedFilter = DependencyFilter.and(specifiedFilter, DependencyFilter.NO_NODEP_ATTRIBUTES);
    }
    return specifiedFilter;
  }

  public static AbstractBlazeQueryEnvironment<Target> newQueryEnvironment(
      TransitivePackageLoader transitivePackageLoader, WalkableGraphFactory graphFactory,
      PackageProvider packageProvider,
      TargetPatternEvaluator targetPatternEvaluator, boolean keepGoing, boolean orderedResults,
      List<String> universeScope, int loadingPhaseThreads,
      EventHandler eventHandler, Set<Setting> settings, Iterable<QueryFunction> functions,
      @Nullable PathPackageLocator packagePath) {
    return newQueryEnvironment(transitivePackageLoader, graphFactory, packageProvider,
        targetPatternEvaluator, keepGoing, /*strictScope=*/true, orderedResults,
        universeScope, loadingPhaseThreads, Rule.ALL_LABELS, eventHandler, settings, functions,
        packagePath);
  }

  public static AbstractBlazeQueryEnvironment<Target> newQueryEnvironment(
      TransitivePackageLoader transitivePackageLoader, WalkableGraphFactory graphFactory,
      PackageProvider packageProvider,
      TargetPatternEvaluator targetPatternEvaluator, boolean keepGoing, boolean strictScope,
      boolean orderedResults, List<String> universeScope, int loadingPhaseThreads,
      Predicate<Label> labelFilter,
      EventHandler eventHandler, Set<Setting> settings, Iterable<QueryFunction> functions,
      @Nullable PathPackageLocator packagePath) {
    Preconditions.checkNotNull(universeScope);
    return orderedResults || universeScope.isEmpty() || packagePath == null
        ? new BlazeQueryEnvironment(transitivePackageLoader, packageProvider,
        targetPatternEvaluator, keepGoing, strictScope, loadingPhaseThreads,
        labelFilter, eventHandler, settings, functions)
        : new SkyQueryEnvironment(
            keepGoing, strictScope, loadingPhaseThreads, labelFilter, eventHandler, settings,
            functions, targetPatternEvaluator.getOffset(), graphFactory, universeScope,
            packagePath);
  }

  /**
   * Evaluate the specified query expression in this environment.
   *
   * @return a {@link QueryEvalResult} object that contains the resulting set of targets and a bit
   *   to indicate whether errors occurred during evaluation; note that the
   *   success status can only be false if {@code --keep_going} was in effect
   * @throws QueryException if the evaluation failed and {@code --nokeep_going} was in
   *   effect
   */
  public QueryEvalResult evaluateQuery(QueryExpression expr, final Callback<T> callback)
      throws QueryException, InterruptedException {

    final AtomicBoolean empty = new AtomicBoolean(true);
    try (final AutoProfiler p = AutoProfiler.logged("evaluating query", LOG)) {

      // In the --nokeep_going case, errors are reported in the order in which the patterns are
      // specified; using a linked hash set here makes sure that the left-most error is reported.
      Set<String> targetPatternSet = new LinkedHashSet<>();
      expr.collectTargetPatterns(targetPatternSet);
      try {
        preloadOrThrow(expr, targetPatternSet);
      } catch (TargetParsingException e) {
        // Unfortunately, by evaluating the patterns in parallel, we lose some location information.
        throw new QueryException(expr, e.getMessage());
      }
      try {
        this.eval(expr, new Callback<T>() {
          @Override
          public void process(Iterable<T> partialResult)
              throws QueryException, InterruptedException {
            empty.compareAndSet(true, Iterables.isEmpty(partialResult));
            callback.process(partialResult);
          }
        });
      } catch (QueryException e) {
        throw new QueryException(e, expr);
      }
    }

    if (eventHandler.hasErrors()) {
      if (!keepGoing) {
        // This case represents loading-phase errors reported during evaluation
        // of target patterns that don't cause evaluation to fail per se.
        throw new QueryException("Evaluation of query \"" + expr
            + "\" failed due to BUILD file errors");
      } else {
        eventHandler.handle(Event.warn("--keep_going specified, ignoring errors.  "
            + "Results may be inaccurate"));
      }
    }

    return new QueryEvalResult(!eventHandler.hasErrors(), empty.get());
  }

  public QueryEvalResult evaluateQuery(String query, Callback<T> callback)
      throws QueryException, InterruptedException {
    return evaluateQuery(QueryExpression.parse(query, this), callback);
  }

  @Override
  public void reportBuildFileError(QueryExpression caller, String message) throws QueryException {
    if (!keepGoing) {
      throw new QueryException(caller, message);
    } else {
      // Keep consistent with evaluateQuery() above.
      eventHandler.handle(Event.error("Evaluation of query \"" + caller + "\" failed: " + message));
    }
  }

  public abstract Target getTarget(Label label) throws TargetNotFoundException, QueryException;

  @Override
  public Set<T> getVariable(String name) {
    return letBindings.get(name);
  }

  @Override
  public Set<T> setVariable(String name, Set<T> value) {
    return letBindings.put(name, value);
  }

  protected boolean validateScope(Label label, boolean strict) throws QueryException {
    if (!labelFilter.apply(label)) {
      String error = String.format("target '%s' is not within the scope of the query", label);
      if (strict) {
        throw new QueryException(error);
      } else {
        eventHandler.handle(Event.warn(error + ". Skipping"));
        return false;
      }
    }
    return true;
  }

  public Set<T> evalTargetPattern(QueryExpression caller, String pattern)
      throws QueryException {
    try {
      preloadOrThrow(caller, ImmutableList.of(pattern));
    } catch (TargetParsingException e) {
      // Will skip the target and keep going if -k is specified.
      reportBuildFileError(caller, e.getMessage());
    }
    AggregateAllCallback<T> aggregatingCallback = new AggregateAllCallback<>();
    getTargetsMatchingPattern(caller, pattern, aggregatingCallback);
    return aggregatingCallback.getResult();
  }

  /**
   * Perform any work that should be done ahead of time to resolve the target patterns in the
   * query. Implementations may choose to cache the results of resolving the patterns, cache
   * intermediate work, or not cache and resolve patterns on the fly.
   */
  protected abstract void preloadOrThrow(QueryExpression caller, Collection<String> patterns)
      throws QueryException, TargetParsingException;

  @Override
  public boolean isSettingEnabled(Setting setting) {
    return settings.contains(Preconditions.checkNotNull(setting));
  }

  @Override
  public Iterable<QueryFunction> getFunctions() {
    ImmutableList.Builder<QueryFunction> builder = ImmutableList.builder();
    builder.addAll(DEFAULT_QUERY_FUNCTIONS);
    builder.addAll(extraFunctions);
    return builder.build();
  }
}
