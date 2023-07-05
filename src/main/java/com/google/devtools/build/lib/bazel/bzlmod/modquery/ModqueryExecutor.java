// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod.modquery;

import static com.google.common.collect.ImmutableSortedMap.toImmutableSortedMap;
import static java.util.Comparator.reverseOrder;
import static java.util.stream.Collectors.toCollection;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode.IsExpanded;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode.IsIndirect;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode.NodeMetadata;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.query2.query.output.BuildOutputFormatter.AttributeReader;
import com.google.devtools.build.lib.query2.query.output.BuildOutputFormatter.TargetOutputter;
import com.google.devtools.build.lib.query2.query.output.PossibleAttributeValues;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.ArrayDeque;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Executes inspection queries for {@link
 * com.google.devtools.build.lib.bazel.commands.ModqueryCommand} and prints the resulted output to
 * the reporter's output stream using the different defined {@link OutputFormatters}.
 */
public class ModqueryExecutor {

  private final ImmutableMap<ModuleKey, AugmentedModule> depGraph;
  private final ModqueryOptions options;
  private final PrintWriter printer;

  public ModqueryExecutor(
      ImmutableMap<ModuleKey, AugmentedModule> depGraph, ModqueryOptions options, Writer writer) {
    this.depGraph = depGraph;
    this.options = options;
    this.printer = new PrintWriter(writer);
  }

  public void tree(ImmutableSet<ModuleKey> targets) {
    ImmutableMap<ModuleKey, ResultNode> result = expandAndPrune(targets, ImmutableSet.of(), false);
    OutputFormatters.getFormatter(options.outputFormat).output(result, depGraph, printer, options);
  }

  public void path(ImmutableSet<ModuleKey> from, ImmutableSet<ModuleKey> to) {
    ImmutableMap<ModuleKey, ResultNode> result = expandAndPrune(from, to, true);
    OutputFormatters.getFormatter(options.outputFormat).output(result, depGraph, printer, options);
  }

  public void allPaths(ImmutableSet<ModuleKey> from, ImmutableSet<ModuleKey> to) {
    ImmutableMap<ModuleKey, ResultNode> result = expandAndPrune(from, to, false);
    OutputFormatters.getFormatter(options.outputFormat).output(result, depGraph, printer, options);
  }

  public void show(ImmutableMap<ModuleKey, BzlmodRepoRuleValue> repoRuleValues) {
    RuleDisplayOutputter outputter = new RuleDisplayOutputter(printer);
    for (Entry<ModuleKey, BzlmodRepoRuleValue> e : repoRuleValues.entrySet()) {
      printer.printf("## %s:", e.getKey());
      outputter.outputRule(e.getValue().getRule());
    }
    printer.flush();
  }

  /**
   * The core function which produces the {@link ResultNode} graph for all the graph-generating
   * queries above. First, it expands the result graph starting from the {@code from} modules, up
   * until the {@code to} target modules if they are specified. If {@code singlePath} is set, it
   * will only contain a single path to one of the targets. <br>
   * Then it calls {@link ResultGraphPruner#pruneByDepth()} to prune nodes after the specified
   * {@code depth} (root is at depth 0). If the query specifies any {@code to} targets, even if they
   * are below the specified depth, they will still be included in the graph using some indirect
   * (dotted) edges. If {@code from} nodes other than the root are specified, they will be pinned
   * (connected directly under the root - using indirect edges if necessary).
   */
  @VisibleForTesting
  ImmutableMap<ModuleKey, ResultNode> expandAndPrune(
      ImmutableSet<ModuleKey> from, ImmutableSet<ModuleKey> to, boolean singlePath) {
    final MaybeCompleteSet<ModuleKey> coloredPaths = colorReversePathsToRoot(to);
    ImmutableMap.Builder<ModuleKey, ResultNode> resultBuilder = new ImmutableMap.Builder<>();

    ImmutableSet<ModuleKey> rootDirectChildren =
        depGraph.get(ModuleKey.ROOT).getAllDeps(options.includeUnused).keySet();
    ImmutableSet<ModuleKey> rootPinnedChildren =
        getPinnedChildrenOfRootInTheResultGraph(rootDirectChildren, from);
    ResultNode.Builder rootBuilder = ResultNode.builder();
    rootPinnedChildren.stream()
        .filter(coloredPaths::contains)
        .forEach(
            moduleKey ->
                rootBuilder.addChild(
                    moduleKey,
                    IsExpanded.TRUE,
                    rootDirectChildren.contains(moduleKey) ? IsIndirect.FALSE : IsIndirect.TRUE));
    resultBuilder.put(ModuleKey.ROOT, rootBuilder.build());

    Set<ModuleKey> seen = new HashSet<>(rootPinnedChildren);
    Deque<ModuleKey> toVisit = new ArrayDeque<>(rootPinnedChildren);
    seen.add(ModuleKey.ROOT);

    while (!toVisit.isEmpty()) {
      ModuleKey key = toVisit.pop();
      AugmentedModule module = depGraph.get(key);
      ResultNode.Builder nodeBuilder = ResultNode.builder();
      nodeBuilder.setTarget(to.contains(key));

      ImmutableSet<ModuleKey> moduleDeps = module.getAllDeps(options.includeUnused).keySet();
      for (ModuleKey childKey : moduleDeps) {
        if (!coloredPaths.contains(childKey)) {
          continue;
        }
        if (to.contains(childKey)) {
          nodeBuilder.setTargetParent(true);
        }
        if (seen.contains(childKey)) {
          // Single paths should not contain cycles or unexpanded (duplicate) children
          // TODO(andreisolo): Move the single path extraction to DFS otherwise it can produce a
          //  wrong answer in cycle edge-case A -> B -> C -> B with target D will not find ABD
          //                                        \__ D
          if (!singlePath) {
            nodeBuilder.addChild(childKey, IsExpanded.FALSE, IsIndirect.FALSE);
          }
          continue;
        }
        nodeBuilder.addChild(childKey, IsExpanded.TRUE, IsIndirect.FALSE);
        seen.add(childKey);
        toVisit.add(childKey);
        if (singlePath) {
          break;
        }
      }

      resultBuilder.put(key, nodeBuilder.build());
    }
    return new ResultGraphPruner(!to.isEmpty(), resultBuilder.buildOrThrow()).pruneByDepth();
  }

  private class ResultGraphPruner {

    private final Map<ModuleKey, ResultNode> oldResult;
    private final Map<ModuleKey, ResultNode.Builder> resultBuilder;
    private final Set<ModuleKey> parentStack;
    private final boolean withTargets;

    /**
     * Prunes the result tree after the specified depth using DFS (because some nodes may still
     * appear after the max depth). <br>
     *
     * @param withTargets If set, it means that the result tree contains paths to some specific
     *     targets. This will cause some branches to contain, after the specified depths, some
     *     targets or target parents. As any other nodes omitted, transitive edges (embedding
     *     multiple edges) will be stored as <i>indirect</i>.
     * @param oldResult The unpruned result graph.
     */
    ResultGraphPruner(boolean withTargets, Map<ModuleKey, ResultNode> oldResult) {
      this.oldResult = oldResult;
      this.resultBuilder = new HashMap<>();
      this.parentStack = new HashSet<>();
      this.withTargets = withTargets;
    }

    private ImmutableMap<ModuleKey, ResultNode> pruneByDepth() {
      ResultNode.Builder rootBuilder = ResultNode.builder();
      resultBuilder.put(ModuleKey.ROOT, rootBuilder);

      parentStack.add(ModuleKey.ROOT);

      for (Entry<ModuleKey, NodeMetadata> e :
          oldResult.get(ModuleKey.ROOT).getChildrenSortedByKey()) {
        rootBuilder.addChild(e.getKey(), IsExpanded.TRUE, e.getValue().isIndirect());
        visitVisible(e.getKey(), 1, ModuleKey.ROOT, IsExpanded.TRUE);
      }

      // Build everything at the end to allow children to add themselves to their parent's
      // adjacency list.
      return resultBuilder.entrySet().stream()
          .collect(
              toImmutableSortedMap(
                  ModuleKey.LEXICOGRAPHIC_COMPARATOR, Entry::getKey, e -> e.getValue().build()));
    }

    // Handles graph traversal within the specified depth.
    private void visitVisible(
        ModuleKey moduleKey, int depth, ModuleKey parentKey, IsExpanded expanded) {
      parentStack.add(moduleKey);
      ResultNode oldNode = oldResult.get(moduleKey);
      ResultNode.Builder nodeBuilder = ResultNode.builder();

      resultBuilder.put(moduleKey, nodeBuilder);
      nodeBuilder.setTarget(oldNode.isTarget());
      if (depth > 1) {
        resultBuilder.get(parentKey).addChild(moduleKey, expanded, IsIndirect.FALSE);
      }

      if (expanded == IsExpanded.FALSE) {
        parentStack.remove(moduleKey);
        return;
      }
      for (Entry<ModuleKey, NodeMetadata> e : oldNode.getChildrenSortedByKey()) {
        ModuleKey childKey = e.getKey();
        IsExpanded childExpanded = e.getValue().isExpanded();
        if (notCycle(childKey)) {
          if (depth < options.depth) {
            visitVisible(childKey, depth + 1, moduleKey, childExpanded);
          } else if (withTargets) {
            visitDetached(childKey, moduleKey, moduleKey, childExpanded);
          }
        } else if (options.cycles) {
          nodeBuilder.addCycle(childKey);
        }
      }
      parentStack.remove(moduleKey);
    }

    // Detached mode is only present in withTargets and handles adding targets and target parents
    // living below the specified depth to the graph.
    private void visitDetached(
        ModuleKey moduleKey,
        ModuleKey parentKey,
        ModuleKey lastVisibleParentKey,
        IsExpanded expanded) {
      parentStack.add(moduleKey);
      ResultNode oldNode = oldResult.get(moduleKey);
      ResultNode.Builder nodeBuilder = ResultNode.builder();
      nodeBuilder.setTarget(oldNode.isTarget());

      if (oldNode.isTarget() || oldNode.isTargetParent()) {
        ResultNode.Builder parentBuilder = resultBuilder.get(lastVisibleParentKey);
        IsIndirect childIndirect =
            lastVisibleParentKey.equals(parentKey) ? IsIndirect.FALSE : IsIndirect.TRUE;
        parentBuilder.addChild(moduleKey, expanded, childIndirect);
        resultBuilder.put(moduleKey, nodeBuilder);
        lastVisibleParentKey = moduleKey;
      }

      if (expanded == IsExpanded.FALSE) {
        parentStack.remove(moduleKey);
        return;
      }
      for (Entry<ModuleKey, NodeMetadata> e : oldNode.getChildrenSortedByKey()) {
        ModuleKey childKey = e.getKey();
        IsExpanded childExpanded = e.getValue().isExpanded();
        if (notCycle(childKey)) {
          visitDetached(childKey, moduleKey, lastVisibleParentKey, childExpanded);
        } else if (options.cycles) {
          nodeBuilder.addCycle(childKey);
        }
      }
      parentStack.remove(moduleKey);
    }

    private boolean notCycle(ModuleKey key) {
      return !parentStack.contains(key);
    }
  }

  /**
   * Return a list of modules that will be the direct children of the root in the result graph
   * (original root's direct dependencies along with the specified targets).
   */
  private ImmutableSet<ModuleKey> getPinnedChildrenOfRootInTheResultGraph(
      ImmutableSet<ModuleKey> rootDirectDeps, ImmutableSet<ModuleKey> fromTargets) {
    Set<ModuleKey> targetKeys =
        fromTargets.stream()
            .filter(k -> filterUnused(k, options.includeUnused, true, depGraph))
            .collect(toCollection(HashSet::new));
    if (fromTargets.contains(ModuleKey.ROOT)) {
      targetKeys.addAll(rootDirectDeps);
    }
    return ImmutableSet.copyOf(targetKeys);
  }

  /**
   * Color all reverse paths from the target modules to the root so only modules which are part of
   * these paths will be included in the output graph during the breadth-first traversal.
   */
  private MaybeCompleteSet<ModuleKey> colorReversePathsToRoot(ImmutableSet<ModuleKey> targets) {
    if (targets.isEmpty()) {
      return MaybeCompleteSet.completeSet();
    }

    Set<ModuleKey> seen = new HashSet<>(targets);
    Deque<ModuleKey> toVisit = new ArrayDeque<>(targets);

    while (!toVisit.isEmpty()) {
      ModuleKey key = toVisit.pop();
      AugmentedModule module = depGraph.get(key);
      Set<ModuleKey> parents = new HashSet<>(module.getDependants());
      if (options.includeUnused) {
        parents.addAll(module.getOriginalDependants());
      }
      for (ModuleKey parent : parents) {
        if (seen.contains(parent)) {
          continue;
        }
        seen.add(parent);
        toVisit.add(parent);
      }
    }

    return MaybeCompleteSet.copyOf(seen);
  }

  /**
   * Helper to filter unused (and unloaded) specified target that cannot be explained and print a
   * warning of that.
   */
  public static boolean filterUnused(
      ModuleKey key,
      boolean includeUnused,
      boolean allowNotLoaded,
      ImmutableMap<ModuleKey, AugmentedModule> dependenciesGraph) {
    AugmentedModule module = Objects.requireNonNull(dependenciesGraph.get(key));
    if (key.equals(ModuleKey.ROOT)) {
      return false;
    }
    if (!module.isUsed() && !includeUnused) {
      return false;
    }
    if (!module.isLoaded()) {
      return allowNotLoaded;
    }
    return true;
  }

  /** A node representing a module that forms the result graph. */
  @AutoValue
  public abstract static class ResultNode {

    /** Whether the module is one of the targets in a paths query. */
    abstract boolean isTarget();

    /**
     * Whether the module is a parent of one of the targets in a paths query (can also be a target).
     */
    abstract boolean isTargetParent();

    enum IsExpanded {
      FALSE,
      TRUE
    }

    enum IsIndirect {
      FALSE,
      TRUE
    }

    enum IsCycle {
      FALSE,
      TRUE
    }

    /** Detailed edge type for the {@link ResultNode} graph. */
    @AutoValue
    public abstract static class NodeMetadata {
      /**
       * Whether the node should be expanded from this edge (the same node can appear in multiple
       * places in a flattened graph).
       */
      public abstract IsExpanded isExpanded();

      /** Whether the edge is a direct edge or an indirect (transitive) one. */
      public abstract IsIndirect isIndirect();

      /** Whether the edge is cycling back inside the flattened graph. */
      public abstract IsCycle isCycle();

      private static NodeMetadata create(
          IsExpanded isExpanded, IsIndirect isIndirect, IsCycle isCycle) {
        return new AutoValue_ModqueryExecutor_ResultNode_NodeMetadata(
            isExpanded, isIndirect, isCycle);
      }
    }

    /** List of children mapped to detailed edge types. */
    protected abstract ImmutableSetMultimap<ModuleKey, NodeMetadata> getChildren();

    public ImmutableSortedSet<Entry<ModuleKey, NodeMetadata>> getChildrenSortedByKey() {
      return ImmutableSortedSet.copyOf(
          Entry.comparingByKey(ModuleKey.LEXICOGRAPHIC_COMPARATOR), getChildren().entries());
    }

    public ImmutableSortedSet<Entry<ModuleKey, NodeMetadata>> getChildrenSortedByEdgeType() {
      return ImmutableSortedSet.copyOf(
          Comparator.<Entry<ModuleKey, NodeMetadata>, IsCycle>comparing(
                  e -> e.getValue().isCycle(), reverseOrder())
              .thenComparing(e -> e.getValue().isExpanded())
              .thenComparing(e -> e.getValue().isIndirect())
              .thenComparing(Entry::getKey, ModuleKey.LEXICOGRAPHIC_COMPARATOR),
          getChildren().entries());
    }

    static ResultNode.Builder builder() {
      return new AutoValue_ModqueryExecutor_ResultNode.Builder()
          .setTarget(false)
          .setTargetParent(false);
    }

    @AutoValue.Builder
    abstract static class Builder {

      abstract ResultNode.Builder setTargetParent(boolean value);

      abstract ResultNode.Builder setTarget(boolean value);

      abstract ImmutableSetMultimap.Builder<ModuleKey, NodeMetadata> childrenBuilder();

      @CanIgnoreReturnValue
      final Builder addChild(ModuleKey value, IsExpanded expanded, IsIndirect indirect) {
        childrenBuilder().put(value, NodeMetadata.create(expanded, indirect, IsCycle.FALSE));
        return this;
      }

      @CanIgnoreReturnValue
      final Builder addCycle(ModuleKey value) {
        childrenBuilder()
            .put(value, NodeMetadata.create(IsExpanded.FALSE, IsIndirect.FALSE, IsCycle.TRUE));
        return this;
      }

      abstract ResultNode build();
    }
  }

  /**
   * A set that either contains some elements or is the <i>complete</i> set (has a special
   * constructor). <br>
   * A complete set is stored internally as {@code null}. However, passing null to {@link
   * #copyOf(Set)} is not allowed. Use {@link #completeSet()} instead.
   */
  @AutoValue
  abstract static class MaybeCompleteSet<T> {
    @Nullable
    protected abstract ImmutableSet<T> internalSet();

    boolean contains(T value) {
      if (internalSet() == null) {
        return true;
      }
      return internalSet().contains(value);
    }

    static <T> MaybeCompleteSet<T> copyOf(Set<T> nullableSet) {
      Preconditions.checkArgument(nullableSet != null);
      return new AutoValue_ModqueryExecutor_MaybeCompleteSet<T>(ImmutableSet.copyOf(nullableSet));
    }

    static <T> MaybeCompleteSet<T> completeSet() {
      return new AutoValue_ModqueryExecutor_MaybeCompleteSet<>(null);
    }
  }

  /**
   * Uses Query's {@link TargetOutputter} to display the generating repo rule and other information.
   */
  static class RuleDisplayOutputter {
    private static final AttributeReader attrReader =
        (rule, attr) ->
            // Query's implementation copied
            PossibleAttributeValues.forRuleAndAttribute(
                rule, attr, /* mayTreatMultipleAsNone= */ true);
    private final TargetOutputter targetOutputter;
    private final PrintWriter printer;

    RuleDisplayOutputter(PrintWriter printer) {
      this.printer = printer;
      this.targetOutputter =
          new TargetOutputter(
              this.printer,
              (rule, attr) -> RawAttributeMapper.of(rule).isConfigurable(attr.getName()),
              "\n");
    }

    private void outputRule(Rule rule) {
      try {
        targetOutputter.outputRule(rule, attrReader, this.printer);
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
    }
  }
}
