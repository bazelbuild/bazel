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
package com.google.devtools.build.lib.query2;

import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.TRISTATE;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.ErrorSensingEventHandler;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.lib.pkgcache.TargetEdgeObserver;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.pkgcache.TargetProvider;
import com.google.devtools.build.lib.query2.engine.BlazeQueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.SkyframeRestartQueryException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.BinaryPredicate;
import com.google.devtools.build.lib.vfs.PathFragment;

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
public class BlazeQueryEnvironment implements QueryEnvironment<Target> {
  protected final ErrorSensingEventHandler eventHandler;
  private final TargetProvider targetProvider;
  private final TargetPatternEvaluator targetPatternEvaluator;
  private final Digraph<Target> graph = new Digraph<>();
  private final ErrorPrintingTargetEdgeErrorObserver errorObserver;
  private final LabelVisitor labelVisitor;
  private final Map<String, Set<Target>> letBindings = new HashMap<>();
  private final Map<String, ResolvedTargets<Target>> resolvedTargetPatterns = new HashMap<>();
  protected final boolean keepGoing;
  private final boolean strictScope;
  protected final int loadingPhaseThreads;

  private final BinaryPredicate<Rule, Attribute> dependencyFilter;
  private final Predicate<Label> labelFilter;

  private final Set<Setting> settings;
  private final List<QueryFunction> extraFunctions;
  private final BlazeTargetAccessor accessor = new BlazeTargetAccessor();

  /**
   * Note that the correct operation of this class critically depends on the Reporter being a
   * singleton object, shared by all cooperating classes contributing to Query.
   * @param strictScope if true, fail the whole query if a label goes out of scope.
   * @param loadingPhaseThreads the number of threads to use during loading
   *     the packages for the query.
   * @param labelFilter a predicate that determines if a specific label is
   *     allowed to be visited during query execution. If it returns false,
   *     the query execution is stopped with an error message.
   * @param settings a set of enabled settings
   */
  public BlazeQueryEnvironment(PackageProvider packageProvider,
      TargetPatternEvaluator targetPatternEvaluator,
      boolean keepGoing,
      boolean strictScope,
      int loadingPhaseThreads,
      Predicate<Label> labelFilter,
      EventHandler eventHandler,
      Set<Setting> settings,
      Iterable<QueryFunction> extraFunctions) {
    this.eventHandler = new ErrorSensingEventHandler(eventHandler);
    this.targetProvider = packageProvider;
    this.targetPatternEvaluator = targetPatternEvaluator;
    this.errorObserver = new ErrorPrintingTargetEdgeErrorObserver(this.eventHandler);
    this.keepGoing = keepGoing;
    this.strictScope = strictScope;
    this.loadingPhaseThreads = loadingPhaseThreads;
    this.dependencyFilter = constructDependencyFilter(settings);
    this.labelVisitor = new LabelVisitor(packageProvider, dependencyFilter);
    this.labelFilter = labelFilter;
    this.settings = Sets.immutableEnumSet(settings);
    this.extraFunctions = ImmutableList.copyOf(extraFunctions);
  }

  /**
   * Note that the correct operation of this class critically depends on the Reporter being a
   * singleton object, shared by all cooperating classes contributing to Query.
   * @param loadingPhaseThreads the number of threads to use during loading
   *     the packages for the query.
   * @param settings a set of enabled settings
   */
  public BlazeQueryEnvironment(PackageProvider packageProvider,
      TargetPatternEvaluator targetPatternEvaluator,
      boolean keepGoing,
      int loadingPhaseThreads,
      EventHandler eventHandler,
      Set<Setting> settings,
      Iterable<QueryFunction> extraFunctions) {
    this(packageProvider, targetPatternEvaluator, keepGoing, /*strictScope=*/true,
        loadingPhaseThreads, Rule.ALL_LABELS, eventHandler, settings, extraFunctions);
  }

  private static BinaryPredicate<Rule, Attribute> constructDependencyFilter(Set<Setting> settings) {
    BinaryPredicate<Rule, Attribute> specifiedFilter =
        settings.contains(Setting.NO_HOST_DEPS) ? Rule.NO_HOST_DEPS : Rule.ALL_DEPS;
    if (settings.contains(Setting.NO_IMPLICIT_DEPS)) {
      specifiedFilter = Rule.and(specifiedFilter, Rule.NO_IMPLICIT_DEPS);
    }
    if (settings.contains(Setting.NO_NODEP_DEPS)) {
      specifiedFilter = Rule.and(specifiedFilter, Rule.NO_NODEP_ATTRIBUTES);
    }
    return specifiedFilter;
  }

  /**
   * Evaluate the specified query expression in this environment.
   *
   * @return a {@link BlazeQueryEvalResult} object that contains the resulting set of targets, the
   *   partial graph, and a bit to indicate whether errors occured during evaluation; note that the
   *   success status can only be false if {@code --keep_going} was in effect
   * @throws QueryException if the evaluation failed and {@code --nokeep_going} was in
   *   effect
   */
  public BlazeQueryEvalResult<Target> evaluateQuery(QueryExpression expr) throws QueryException {
    // Some errors are reported as QueryExceptions and others as ERROR events
    // (if --keep_going).
    eventHandler.resetErrors();
    resolvedTargetPatterns.clear();

    // In the --nokeep_going case, errors are reported in the order in which the patterns are
    // specified; using a linked hash set here makes sure that the left-most error is reported.
    Set<String> targetPatternSet = new LinkedHashSet<>();
    expr.collectTargetPatterns(targetPatternSet);
    try {
      resolvedTargetPatterns.putAll(preloadOrThrow(targetPatternSet));
    } catch (TargetParsingException e) {
      // Unfortunately, by evaluating the patterns in parallel, we lose some location information.
      throw new QueryException(expr, e.getMessage());
    }

    Set<Target> resultNodes;
    try {
      resultNodes = expr.eval(this);
    } catch (QueryException e) {
      throw new QueryException(e, expr);
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

    return new BlazeQueryEvalResult<>(!eventHandler.hasErrors(), resultNodes, graph);
  }

  public BlazeQueryEvalResult<Target> evaluateQuery(String query) throws QueryException {
    return evaluateQuery(QueryExpression.parse(query, this));
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

  @Override
  public Set<Target> getTargetsMatchingPattern(QueryExpression caller,
      String pattern) throws QueryException {
    // We can safely ignore the boolean error flag. The evaluateQuery() method above wraps the
    // entire query computation in an error sensor.

    Set<Target> targets = new LinkedHashSet<>(resolvedTargetPatterns.get(pattern).getTargets());

    // Sets.filter would be more convenient here, but can't deal with exceptions.
    Iterator<Target> targetIterator = targets.iterator();
    while (targetIterator.hasNext()) {
      Target target = targetIterator.next();
      if (!validateScope(target.getLabel(), strictScope)) {
        targetIterator.remove();
      }
    }

    Set<PathFragment> packages = new HashSet<>();
    for (Target target : targets) {
      packages.add(target.getLabel().getPackageFragment());
    }

    Set<Target> result = new LinkedHashSet<>();
    for (Target target : targets) {
      result.add(getOrCreate(target));

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
          } catch (InterruptedException e) {
            throw new QueryException("interrupted");
          }
        }
      }
    }
    return result;
  }

  public Target getTarget(Label label) throws TargetNotFoundException, QueryException {
    // Can't use strictScope here because we are expecting a target back.
    validateScope(label, true);
    try {
      return getNode(getTargetOrThrow(label)).getLabel();
    } catch (NoSuchThingException e) {
      throw new TargetNotFoundException(e);
    } catch (InterruptedException e) {
      throw new QueryException("interrupted");
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
  public Collection<Target> getFwdDeps(Target target) {
    return getTargetsFromNodes(getNode(target).getSuccessors());
  }

  @Override
  public Collection<Target> getReverseDeps(Target target) {
    return getTargetsFromNodes(getNode(target).getPredecessors());
  }

  @Override
  public Set<Target> getTransitiveClosure(Set<Target> targetNodes) {
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

  protected void preloadTransitiveClosure(Set<Target> targets, int maxDepth) throws QueryException {
  }

  @Override
  public void buildTransitiveClosure(QueryExpression caller,
                                     Set<Target> targetNodes,
                                     int maxDepth) throws QueryException {
    Set<Target> targets = targetNodes;
    preloadTransitiveClosure(targets, maxDepth);

    try {
      labelVisitor.syncWithVisitor(eventHandler, targets, keepGoing,
          loadingPhaseThreads, maxDepth, errorObserver, new GraphBuildingObserver());
    } catch (InterruptedException e) {
      throw new QueryException(caller, "transitive closure computation was interrupted");
    }

    if (errorObserver.hasErrors()) {
      reportBuildFileError(caller, "errors were encountered while computing transitive closure");
    }
  }

  @Override
  public Set<Target> getNodesOnPath(Target from, Target to) {
    return getTargetsFromNodes(graph.getShortestPath(getNode(from), getNode(to)));
  }

  @Override
  public Set<Target> getVariable(String name) {
    return letBindings.get(name);
  }

  @Override
  public Set<Target> setVariable(String name, Set<Target> value) {
    return letBindings.put(name, value);
  }

  /**
   * It suffices to synchronize the modifications of this.graph from within the
   * GraphBuildingObserver, because that's the only concurrent part.
   * Concurrency is always encapsulated within the evaluation of a single query
   * operator (e.g. deps(), somepath(), etc).
   */
  private class GraphBuildingObserver implements TargetEdgeObserver {

    @Override
    public synchronized void edge(Target from, Attribute attribute, Target to) {
      Preconditions.checkState(attribute == null ||
          dependencyFilter.apply(((Rule) from), attribute),
          "Disallowed edge from LabelVisitor: %s --> %s", from, to);
      makeEdge(from, to);
    }

    @Override
    public synchronized void node(Target node) {
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

  private boolean validateScope(Label label, boolean strict) throws QueryException {
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

  public Set<Target> evalTargetPattern(QueryExpression caller, String pattern)
      throws QueryException {
    if (!resolvedTargetPatterns.containsKey(pattern)) {
      try {
        resolvedTargetPatterns.putAll(preloadOrThrow(ImmutableList.of(pattern)));
      } catch (TargetParsingException e) {
        // Will skip the target and keep going if -k is specified.
        resolvedTargetPatterns.put(pattern, ResolvedTargets.<Target>empty());
        reportBuildFileError(caller, e.getMessage());
      }
    }
    return getTargetsMatchingPattern(caller, pattern);
  }

  private Map<String, ResolvedTargets<Target>> preloadOrThrow(Collection<String> patterns)
      throws TargetParsingException {
    try {
      // Note that this may throw a RuntimeException if deps are missing in Skyframe.
      return targetPatternEvaluator.preloadTargetPatterns(
          eventHandler, patterns, keepGoing);
    } catch (InterruptedException e) {
      // TODO(bazel-team): Propagate the InterruptedException from here [skyframe-loading].
      throw new TargetParsingException("interrupted");
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

  // TODO(bazel-team): rename this to getDependentFiles when all implementations
  // of QueryEnvironment is fixed.
  @Override
  public Set<Target> getBuildFiles(final QueryExpression caller, Set<Target> nodes)
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
        addIfUniqueLabel(getNode(pkg.getBuildFile()), seenLabels, dependentFiles);
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

  private static void addIfUniqueLabel(Node<Target> node, Set<Label> labels, Set<Target> nodes) {
    if (labels.add(node.getLabel().getLabel())) {
      nodes.add(node.getLabel());
    }
  }

  private Node<Target> getSubincludeTarget(final Label label, Package pkg) {
    return getNode(new FakeSubincludeTarget(label, pkg.getBuildFile().getLocation()));
  }

  @Override
  public TargetAccessor<Target> getAccessor() {
    return accessor;
  }

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

  private final class BlazeTargetAccessor implements TargetAccessor<Target> {

    @Override
    public String getTargetKind(Target target) {
      return target.getTargetKind();
    }

    @Override
    public String getLabel(Target target) {
      return target.getLabel().toString();
    }

    @Override
    public List<Target> getLabelListAttr(QueryExpression caller, Target target, String attrName,
        String errorMsgPrefix) throws QueryException {
      Preconditions.checkArgument(target instanceof Rule);

      List<Target> result = new ArrayList<>();
      Rule rule = (Rule) target;

      AggregatingAttributeMapper attrMap = AggregatingAttributeMapper.of(rule);
      Type<?> attrType = attrMap.getAttributeType(attrName);
      if (attrType == null) {
        // Return an empty list if the attribute isn't defined for this rule.
        return ImmutableList.of();
      }
      for (Object value : attrMap.visitAttribute(attrName, attrType)) {
        // Computed defaults may have null values.
        if (value != null) {
          for (Label label : attrType.getLabels(value)) {
            try {
              result.add(getTarget(label));
            } catch (TargetNotFoundException e) {
              reportBuildFileError(caller, errorMsgPrefix + e.getMessage());
            }
          }
        }
      }

      return result;
    }

    @Override
    public List<String> getStringListAttr(Target target, String attrName) {
      Preconditions.checkArgument(target instanceof Rule);
      return NonconfigurableAttributeMapper.of((Rule) target).get(attrName, Type.STRING_LIST);
    }

    @Override
    public String getStringAttr(Target target, String attrName) {
      Preconditions.checkArgument(target instanceof Rule);
      return NonconfigurableAttributeMapper.of((Rule) target).get(attrName, Type.STRING);
    }

    @Override
    public Iterable<String> getAttrAsString(Target target, String attrName) {
      Preconditions.checkArgument(target instanceof Rule);
      List<String> values = new ArrayList<>(); // May hold null values.
      Attribute attribute = ((Rule) target).getAttributeDefinition(attrName);
      if (attribute != null) {
        Type<?> attributeType = attribute.getType();
        for (Object attrValue : AggregatingAttributeMapper.of((Rule) target).visitAttribute(
            attribute.getName(), attributeType)) {

          // Ugly hack to maintain backward 'attr' query compatibility for BOOLEAN and TRISTATE
          // attributes. These are internally stored as actual Boolean or TriState objects but were
          // historically queried as integers. To maintain compatibility, we inspect their actual
          // value and return the integer equivalent represented as a String. This code is the
          // opposite of the code in BooleanType and TriStateType respectively.
          if (attributeType == BOOLEAN) {
            values.add(Type.BOOLEAN.cast(attrValue) ? "1" : "0");
          } else if (attributeType == TRISTATE) {
              switch (Type.TRISTATE.cast(attrValue)) {
                case AUTO :
                  values.add("-1");
                  break;
                case NO :
                  values.add("0");
                  break;
                case YES :
                  values.add("1");
                  break;
                default :
                  throw new AssertionError("This can't happen!");
              }
          } else {
            values.add(attrValue == null ? null : attrValue.toString());
          }
        }
      }
      return values;
    }

    @Override
    public boolean isRule(Target target) {
      return target instanceof Rule;
    }

    @Override
    public boolean isTestRule(Target target) {
      return TargetUtils.isTestRule(target);
    }

    @Override
    public boolean isTestSuite(Target target) {
      return TargetUtils.isTestSuiteRule(target);
    }
  }

  /** Given a set of target nodes, returns the targets. */
  private static Set<Target> getTargetsFromNodes(Iterable<Node<Target>> input) {
    Set<Target> result = new LinkedHashSet<>();
    for (Node<Target> node : input) {
      result.add(node.getLabel());
    }
    return result;
  }
}
