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

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.graph.Digraph;
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

import javax.annotation.Nullable;

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

  private static SkyKey transformToKey(Target value) {
    return TransitiveTargetValue.key(value.getLabel());
  }

  @Nullable
  private Target transformToValue(SkyKey key) {
    SkyFunctionName functionName = key.functionName();
    if (functionName != SkyFunctions.TRANSITIVE_TARGET) {
      return null;
    }
    try {
      return getTarget(((Label) key.argument()));
    } catch (QueryException | TargetNotFoundException e) {
      // Any problems with targets were already reported during #buildTransitiveClosure.
      return null;
    }
  }

  private Collection<Target> getRawFwdDeps(Target target) {
    return makeTargets(graph.getDirectDeps(transformToKey(target)));
  }

  private Collection<Target> getRawReverseDeps(Target target) {
    return makeTargets(graph.getReverseDeps(transformToKey(target)));
  }

  private Set<Label> getAllowedDeps(Rule rule) {
    Set<Label> allowedLabels = new HashSet<>(rule.getLabels(dependencyFilter));
    allowedLabels.addAll(rule.getVisibility().getDependencyLabels());
    // We should add deps from aspects, otherwise they are going to be filtered out.
    allowedLabels.addAll(rule.getAspectLabelsSuperset(dependencyFilter));
    return allowedLabels;
  }

  public Collection<Target> getFwdDeps(Target target) {
    Collection<Target> unfilteredDeps = getRawFwdDeps(target);
    if (!(target instanceof Rule)) {
      return getRawFwdDeps(target);
    }
    final Set<Label> allowedLabels = getAllowedDeps((Rule) target);
    return Collections2.filter(unfilteredDeps,
        new Predicate<Target>() {
          @Override
          public boolean apply(Target target) {
            return allowedLabels.contains(target.getLabel());
          }
        });
  }

  public Collection<Target> getReverseDeps(final Target target) {
    return Collections2.filter(getRawReverseDeps(target), new Predicate<Target>() {
      @Override
      public boolean apply(Target parent) {
        return !(parent instanceof Rule)
            || getAllowedDeps((Rule) parent).contains(target.getLabel());
      }
    });
  }

  @Override
  public Set<Target> getTransitiveClosure(Set<Target> targets) {
    Set<Target> visited = new HashSet<>();
    List<Target> result = new ArrayList<>(targets);
    int i = 0;
    while (i < result.size()) {
      for (Target dep : getFwdDeps(result.get(i))) {
        if (visited.add(dep)) {
          result.add(dep);
        }
      }
      i++;
    }
    return ImmutableSet.copyOf(result);
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
      for (Target dep : getFwdDeps(current)) {
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
    return new FakeSubincludeTarget(label, pkg.getBuildFile().getLocation());
  }

  @Override
  public TargetAccessor<Target> getAccessor() {
    return accessor;
  }

  @Override
  public Target getTarget(Label label) throws TargetNotFoundException, QueryException {
    // Can't use strictScope here because we are expecting a target back.
    validateScope(label, true);
    SkyKey packageKey = PackageValue.key(label.getPackageIdentifier());
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

  protected Map<String, ResolvedTargets<Target>> preloadOrThrow(QueryExpression caller,
      Collection<String> patterns) throws QueryException, TargetParsingException {
    Map<String, ResolvedTargets<Target>> result = Maps.newHashMapWithExpectedSize(patterns.size());
    for (String pattern : patterns) {
      SkyKey patternKey = TargetPatternValue.key(pattern,
          TargetPatternEvaluator.DEFAULT_FILTERING_POLICY, parserPrefix);

      TargetPatternValue.TargetPattern targetPattern =
          ((TargetPatternValue.TargetPattern) patternKey.argument());

      TargetParsingException targetParsingException = null;
      if (graph.exists(patternKey)) {
        // If the graph already contains a value for this target pattern, use it.
        TargetPatternValue value = (TargetPatternValue) graph.getValue(patternKey);
        if (value != null) {
          result.put(pattern, value.getTargets());
        } else {
          targetParsingException =
              (TargetParsingException)
                  Preconditions.checkNotNull(graph.getException(patternKey), pattern);
        }
      } else {
        // If the graph doesn't contain a value for this target pattern, try to directly evaluate
        // it, by making use of packages already present in the graph.
        TargetPattern.Parser parser = new TargetPattern.Parser(targetPattern.getOffset());
        GraphBackedRecursivePackageProvider provider =
            new GraphBackedRecursivePackageProvider(graph);
        RecursivePackageProviderBackedTargetPatternResolver resolver =
            new RecursivePackageProviderBackedTargetPatternResolver(provider, eventHandler,
                targetPattern.getPolicy(), pkgPath);
        TargetPattern parsedPattern = parser.parse(targetPattern.getPattern());
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

  private Set<Target> makeTargets(Iterable<SkyKey> keys) {
    ImmutableSet.Builder<Target> builder = ImmutableSet.builder();
    for (SkyKey key : keys) {
      Target value = transformToValue(key);
      if (value != null) {
        // Some values may be filtered out because they are not Targets.
        builder.add(value);
      }
    }
    return builder.build();
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
