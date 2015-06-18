// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Supplier;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multimaps;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.skyframe.GraphBackedRecursivePackageProvider;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.RecursivePackageProviderBackedTargetPatternResolver;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.TargetPatternValue;
import com.google.devtools.build.lib.skyframe.TransitiveTargetValue;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.devtools.build.skyframe.WalkableGraph.WalkableGraphFactory;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * {@link AbstractBlazeQueryEnvironment} that introspects the Skyframe graph to find forward and
 * reverse edges. Results obtained by calling {@link #evaluateQuery} are not guaranteed to be in
 * any particular order. As well, this class eagerly loads the full transitive closure of targets,
 * even if the full closure isn't needed.
 */
public class SkyQueryEnvironment extends AbstractBlazeQueryEnvironment<Target> {
  private WalkableGraph graph;

  private final BlazeTargetAccessor accessor = new BlazeTargetAccessor(this);
  private final int loadingPhaseThreads;
  private final WalkableGraphFactory graphFactory;
  private final List<String> universeScope;
  private final String parserPrefix;
  private final PathPackageLocator pkgPath;

  public SkyQueryEnvironment(boolean keepGoing, boolean strictScope, int loadingPhaseThreads,
      Predicate<Label> labelFilter,
      EventHandler eventHandler,
      Set<Setting> settings,
      Iterable<QueryFunction> extraFunctions, String parserPrefix,
      WalkableGraphFactory graphFactory,
      List<String> universeScope, PathPackageLocator pkgPath) {
    super(keepGoing, strictScope, labelFilter,
        eventHandler,
        settings,
        extraFunctions);
    this.loadingPhaseThreads = loadingPhaseThreads;
    this.graphFactory = graphFactory;
    this.pkgPath = pkgPath;
    this.universeScope = Preconditions.checkNotNull(universeScope);
    this.parserPrefix = parserPrefix;
    Preconditions.checkState(!universeScope.isEmpty(),
        "No queries can be performed with an empty universe");
  }

  private void init() throws InterruptedException {
    graph = graphFactory.prepareAndGet(universeScope, loadingPhaseThreads, eventHandler);
  }

  @Override
  public QueryEvalResult<Target> evaluateQuery(QueryExpression expr)
      throws QueryException {
    // Some errors are reported as QueryExceptions and others as ERROR events (if --keep_going). The
    // result is set to have an error iff there were errors emitted during the query, so we reset
    // errors here.
    eventHandler.resetErrors();
    try {
      init();
    } catch (InterruptedException e) {
      throw new QueryException(e.getMessage());
    }
    return super.evaluateQuery(expr);
  }

  private static SkyKey makeKey(Target value) {
    return TransitiveTargetValue.key(value.getLabel());
  }

  private Map<Target, Collection<Target>> makeTargetsMap(Map<SkyKey, Iterable<SkyKey>> input) {
    ImmutableMap.Builder<Target, Collection<Target>> result = ImmutableMap.builder();

    for (Map.Entry<SkyKey, Target> entry : makeTargetsWithAssociations(input.keySet()).entrySet()) {
      result.put(entry.getValue(), makeTargets(input.get(entry.getKey())));
    }
    return result.build();
  }

  private Map<Target, Collection<Target>> getRawFwdDeps(Iterable<Target> targets) {
    return makeTargetsMap(graph.getDirectDeps(makeKeys(targets)));
  }

  private Map<Target, Collection<Target>> getRawReverseDeps(Iterable<Target> targets) {
    return makeTargetsMap(graph.getReverseDeps(makeKeys(targets)));
  }

  private Set<Label> getAllowedDeps(Rule rule) {
    Set<Label> allowedLabels = new HashSet<>(rule.getLabels(dependencyFilter));
    allowedLabels.addAll(rule.getVisibility().getDependencyLabels());
    // We should add deps from aspects, otherwise they are going to be filtered out.
    allowedLabels.addAll(rule.getAspectLabelsSuperset(dependencyFilter));
    return allowedLabels;
  }

  private Collection<Target> filterFwdDeps(Target target, Collection<Target> rawFwdDeps) {
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
  public Collection<Target> getFwdDeps(Iterable<Target> targets) {
    Set<Target> result = new HashSet<>();
    for (Map.Entry<Target, Collection<Target>> entry : getRawFwdDeps(targets).entrySet()) {
      result.addAll(filterFwdDeps(entry.getKey(), entry.getValue()));
    }
    return result;
  }

  private Collection<Target> filterReverseDeps(final Target target,
      Collection<Target> rawReverseDeps) {
    return Collections2.filter(rawReverseDeps, new Predicate<Target>() {
      @Override
      public boolean apply(Target parent) {
        return !(parent instanceof Rule)
            || getAllowedDeps((Rule) parent).contains(target.getLabel());
      }
    });

  }

  @Override
  public Collection<Target> getReverseDeps(Iterable<Target> targets) {
    Set<Target> result = new HashSet<>();
    for (Map.Entry<Target, Collection<Target>> entry : getRawReverseDeps(targets).entrySet()) {
      result.addAll(filterReverseDeps(entry.getKey(), entry.getValue()));
    }
    return result;
  }

  @Override
  public Set<Target> getTransitiveClosure(Set<Target> targets) {
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
  public Set<Target> getNodesOnPath(Target from, Target to) {
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

  @Override
  public Set<Target> getTargetsMatchingPattern(QueryExpression owner, String pattern)
      throws QueryException {
    Set<Target> targets = new LinkedHashSet<>(resolvedTargetPatterns.get(pattern).getTargets());

    // Sets.filter would be more convenient here, but can't deal with exceptions.
    Iterator<Target> targetIterator = targets.iterator();
    while (targetIterator.hasNext()) {
      Target target = targetIterator.next();
      if (!validateScope(target.getLabel(), strictScope)) {
        targetIterator.remove();
      }
    }
    return targets;
  }

  @Override
  public Set<Target> getBuildFiles(QueryExpression caller, Set<Target> nodes)
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
        addIfUniqueLabel(pkg.getBuildFile(), seenLabels, dependentFiles);
        for (Label subinclude
            : Iterables.concat(pkg.getSubincludeLabels(), pkg.getSkylarkFileDependencies())) {
          addIfUniqueLabel(getSubincludeTarget(subinclude, pkg), seenLabels, dependentFiles);

          // Also add the BUILD file of the subinclude.
          try {
            addIfUniqueLabel(getSubincludeTarget(
                subinclude.getLocalTargetLabel("BUILD"), pkg), seenLabels, dependentFiles);
          } catch (Label.SyntaxException e) {
            throw new AssertionError("BUILD should always parse as a target name", e);
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

  private static Target getSubincludeTarget(final Label label, Package pkg) {
    return new FakeSubincludeTarget(label, pkg);
  }

  @Override
  public TargetAccessor<Target> getAccessor() {
    return accessor;
  }

  private SkyKey getPackageKeyAndValidateLabel(Label label) throws QueryException {
    // Can't use strictScope here because we are expecting a target back.
    validateScope(label, true);
    return PackageValue.key(label.getPackageIdentifier());
  }

  @Override
  public Target getTarget(Label label) throws TargetNotFoundException, QueryException {
    SkyKey packageKey = getPackageKeyAndValidateLabel(label);
    checkExistence(packageKey);
    try {
      PackageValue packageValue =
          (PackageValue) graph.getValue(packageKey);
      if (packageValue != null) {
        return packageValue.getPackage().getTarget(label.getName());
      } else {
        throw (NoSuchThingException) Preconditions.checkNotNull(
            graph.getException(packageKey), label);
      }
    } catch (NoSuchThingException e) {
      throw new TargetNotFoundException(e);
    }
  }

  @Override
  public void buildTransitiveClosure(QueryExpression caller, Set<Target> targets, int maxDepth)
      throws QueryException {
    // Everything has already been loaded, so here we just check for errors so that we can
    // pre-emptively throw/report if needed.
    for (Target target : targets) {
      SkyKey targetKey = TransitiveTargetValue.key(target.getLabel());
      checkExistence(targetKey);
      Exception exception = graph.getException(targetKey);
      if (exception != null) {
        reportBuildFileError(caller, exception.getMessage());
      }
    }
  }

  private static Target getExistingTarget(Label label,
      GraphBackedRecursivePackageProvider provider) {
    StoredEventHandler handler = new StoredEventHandler();
    try {
      return provider.getTarget(handler, label);
    } catch (NoSuchThingException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  protected Map<String, ResolvedTargets<Target>> preloadOrThrow(QueryExpression caller,
      Collection<String> patterns) throws QueryException, TargetParsingException {
    GraphBackedRecursivePackageProvider provider =
        new GraphBackedRecursivePackageProvider(graph);
    Map<String, ResolvedTargets<Target>> result = Maps.newHashMapWithExpectedSize(patterns.size());
    for (String pattern : patterns) {
      SkyKey patternKey = TargetPatternValue.key(pattern,
          TargetPatternEvaluator.DEFAULT_FILTERING_POLICY, parserPrefix);

      TargetPatternValue.TargetPatternKey targetPatternKey =
          ((TargetPatternValue.TargetPatternKey) patternKey.argument());

      TargetParsingException targetParsingException = null;
      if (graph.exists(patternKey)) {
        // The graph already contains a value or exception for this target pattern, so we use it.
        TargetPatternValue value = (TargetPatternValue) graph.getValue(patternKey);
        if (value != null) {
          ResolvedTargets.Builder<Target> targetsBuilder = ResolvedTargets.builder();
          for (Label label : value.getTargets().getTargets()) {
            targetsBuilder.add(getExistingTarget(label, provider));
          }
          for (Label label : value.getTargets().getFilteredTargets()) {
            targetsBuilder.remove(getExistingTarget(label, provider));
          }
          result.put(pattern, targetsBuilder.build());
        } else {
          // Because the graph was always initialized via a keep_going build, we know that the
          // exception stored here must be a TargetParsingException. Thus the comment in
          // SkyframeTargetPatternEvaluator#parseTargetPatternKeys describing the situation in which
          // the exception acceptance must be looser does not apply here.
          targetParsingException =
              (TargetParsingException)
                  Preconditions.checkNotNull(graph.getException(patternKey), pattern);
        }
      } else {
        // If the graph doesn't contain a value for this target pattern, try to directly evaluate
        // it, by making use of packages already present in the graph.
        RecursivePackageProviderBackedTargetPatternResolver resolver =
            new RecursivePackageProviderBackedTargetPatternResolver(provider, eventHandler,
                targetPatternKey.getPolicy(), pkgPath);
        TargetPattern parsedPattern = targetPatternKey.getParsedPattern();
        try {
          result.put(pattern, parsedPattern.eval(resolver));
        } catch (TargetParsingException e) {
          targetParsingException = e;
        } catch (InterruptedException e) {
          throw new QueryException(e.getMessage());
        }
      }

      if (targetParsingException != null) {
        if (!keepGoing) {
          throw targetParsingException;
        } else {
          eventHandler.handle(Event.error("Evaluation of query \"" + caller + "\" failed: "
              + targetParsingException.getMessage()));
          result.put(pattern, ResolvedTargets.<Target>builder().setError().build());
        }
      }
    }
    return result;
  }

  private Collection<Target> makeTargets(Iterable<SkyKey> keys) {
    return makeTargetsWithAssociations(keys).values();
  }

  private Map<SkyKey, Target> makeTargetsWithAssociations(Iterable<SkyKey> keys) {
    Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap = Multimaps.newListMultimap(
        new HashMap<SkyKey, Collection<SkyKey>>(),
        new Supplier<List<SkyKey>>() {
          @Override
          public List<SkyKey> get() {
            return new ArrayList<>();
          }
        });
    for (SkyKey key : keys) {
      SkyFunctionName functionName = key.functionName();
      if (!functionName.equals(SkyFunctions.TRANSITIVE_TARGET)) {
        // Skip non-targets.
        continue;
      }
      try {
        packageKeyToTargetKeyMap.put(getPackageKeyAndValidateLabel((Label) key.argument()), key);
      } catch (QueryException e) {
        // Skip disallowed labels.
      }
    }
    ImmutableMap.Builder<SkyKey, Target> result = ImmutableMap.builder();
    Map<SkyKey, SkyValue> packageMap = graph.getValuesMaybe(packageKeyToTargetKeyMap.keySet());
    for (Map.Entry<SkyKey, SkyValue> entry : packageMap.entrySet()) {
      for (SkyKey targetKey : packageKeyToTargetKeyMap.get(entry.getKey())) {
        try {
          result.put(targetKey, ((PackageValue) entry.getValue()).getPackage()
              .getTarget(((Label) targetKey.argument()).getName()));
        } catch (NoSuchTargetException e) {
          // Skip missing target.
        }
      }
    }
    return result.build();
  }

  private Iterable<SkyKey> makeKeys(Iterable<Target> targets) {
    return Iterables.transform(targets, new Function<Target, SkyKey>() {
      @Override
      public SkyKey apply(Target target) {
        return TransitiveTargetValue.key(target.getLabel());
      }
    });
  }

  private void checkExistence(SkyKey key) throws QueryException {
    if (!graph.exists(key)) {
      throw new QueryException(key + " does not exist in graph");
    }
  }

  @Override
  public Target getOrCreate(Target target) {
    return target;
  }
}
