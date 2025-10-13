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

package com.google.devtools.build.lib.bazel.bzlmod.modcommand;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.collect.ImmutableSortedMap.toImmutableSortedMap;
import static com.google.common.collect.ImmutableSortedSet.toImmutableSortedSet;
import static java.util.Comparator.reverseOrder;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.bazel.bzlmod.AttributeValues;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionId;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionUsage;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.Tag;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModExecutor.ResultNode.IsExpanded;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModExecutor.ResultNode.IsIndirect;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModExecutor.ResultNode.NodeMetadata;
import com.google.devtools.build.lib.bazel.repository.RepoDefinition;
import com.google.devtools.build.lib.bazel.repository.RepoRule;
import com.google.devtools.build.lib.util.MaybeCompleteSet;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import net.starlark.java.eval.Starlark;

/**
 * Executes inspection queries for {@link com.google.devtools.build.lib.bazel.commands.ModCommand}
 * and prints the resulted output to the reporter's output stream using the different defined {@link
 * OutputFormatters}.
 */
public class ModExecutor {

  private final ImmutableMap<ModuleKey, AugmentedModule> depGraph;
  private final ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage> extensionUsages;
  private final ImmutableSetMultimap<ModuleExtensionId, String> extensionRepos;
  private final Optional<MaybeCompleteSet<ModuleExtensionId>> extensionFilter;
  private final ModOptions options;
  private final PrintWriter printer;
  private ImmutableMap<ModuleExtensionId, ImmutableSetMultimap<String, ModuleKey>>
      extensionRepoImports;

  public ModExecutor(
      ImmutableMap<ModuleKey, AugmentedModule> depGraph, ModOptions options, Writer writer) {
    this(
        depGraph,
        ImmutableTable.of(),
        ImmutableSetMultimap.of(),
        Optional.of(MaybeCompleteSet.completeSet()),
        options,
        writer);
  }

  public ModExecutor(
      ImmutableMap<ModuleKey, AugmentedModule> depGraph,
      ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage> extensionUsages,
      ImmutableSetMultimap<ModuleExtensionId, String> extensionRepos,
      Optional<MaybeCompleteSet<ModuleExtensionId>> extensionFilter,
      ModOptions options,
      Writer writer) {
    this.depGraph = depGraph;
    this.extensionUsages = extensionUsages;
    this.extensionRepos = extensionRepos;
    this.extensionFilter = extensionFilter;
    this.options = options;
    this.printer = new PrintWriter(writer);
    // Easier lookup table for repo imports by module.
    // It is updated after pruneByDepthAndLink to filter out pruned modules.
    this.extensionRepoImports = computeRepoImportsTable(depGraph.keySet());
  }

  public void graph(ImmutableSet<ModuleKey> from) {
    ImmutableMap<ModuleKey, ResultNode> result;
    ImmutableSet<ModuleKey> targets = computeExtensionFilterTargets();
    if (targets.isEmpty()) {
      result = expandAndPrune(from);
    } else {
      result = expandPathsToTargets(from, targets, false);
    }
    OutputFormatters.getFormatter(options.outputFormat)
        .output(result, depGraph, extensionRepos, extensionRepoImports, printer, options);
  }

  public void path(ImmutableSet<ModuleKey> from, ImmutableSet<ModuleKey> to) {
    ImmutableSet<ModuleKey> targets =
        ImmutableSet.<ModuleKey>builder()
            .addAll(computeExtensionFilterTargets())
            .addAll(to)
            .build();

    if (targets.isEmpty()) {
      printer.println("No target modules specified.");
      printer.flush();
      return;
    }

    ImmutableMap<ModuleKey, ResultNode> result = expandPathsToTargets(from, targets, true);
    if (result.isEmpty()) {
      printer.println("No path found to the specified target modules.");
      printer.flush();
      return;
    }
    OutputFormatters.getFormatter(options.outputFormat)
        .output(result, depGraph, extensionRepos, extensionRepoImports, printer, options);
  }

  public void allPaths(ImmutableSet<ModuleKey> from, ImmutableSet<ModuleKey> to) {
    ImmutableSet<ModuleKey> targets =
        ImmutableSet.<ModuleKey>builder()
            .addAll(computeExtensionFilterTargets())
            .addAll(to)
            .build();

    if (targets.isEmpty()) {
      printer.println("No target modules specified.");
      printer.flush();
      return;
    }

    ImmutableMap<ModuleKey, ResultNode> result = expandPathsToTargets(from, targets, false);
    if (result.isEmpty()) {
      printer.println("No path found to the specified target modules.");
      printer.flush();
      return;
    }
    OutputFormatters.getFormatter(options.outputFormat)
        .output(result, depGraph, extensionRepos, extensionRepoImports, printer, options);
  }

  public void showRepo(ImmutableMap<String, RepoDefinition> targetRepoDefinitions) {
    for (Map.Entry<String, RepoDefinition> e : targetRepoDefinitions.entrySet()) {
      printer.printf("## %s:\n", e.getKey());
      printRepoDefinition(e.getValue());
    }
    printer.flush();
  }

  public void showExtension(
      ImmutableSet<ModuleExtensionId> extensions, ImmutableSet<ModuleKey> fromUsages)
      throws InvalidArgumentException {
    for (ModuleExtensionId extension : extensions) {
      displayExtension(extension, fromUsages);
    }
    printer.flush();
  }

  /**
   * Reconstructs a path backwards from a child to the root and adds it to the result graph.
   *
   * <p>This is a helper function for {@link #expandPathsToTargets}. Once a path to a target is
   * found, this function is called to walk up the dependency chain (using the {@code bfsParentMap})
   * and add the necessary nodes and edges to the {@code resultGraph}.
   */
  private void addPathToResultGraph(
      Map<ModuleKey, ResultNode> resultGraph,
      Map<ModuleKey, ModuleKey> bfsParentMap,
      ModuleKey pathParent,
      ModuleKey pathChild) {
    // Mark the child node as a target in the result graph.
    ResultNode.Builder childNodeBuilder = ResultNode.builder();
    if (resultGraph.containsKey(pathChild)) {
      childNodeBuilder.addChildren(resultGraph.get(pathChild).getChildren());
    }
    resultGraph.put(pathChild, childNodeBuilder.setTarget(true).build());

    // Traverse up from the found path to the root, adding the path to the result graph.
    ImmutableSortedSet<ModuleKey> rootDirectChildren =
        depGraph.get(ModuleKey.ROOT).getAllDeps(options.includeUnused).keySet();

    ModuleKey currentChild = pathChild;
    ModuleKey currentParent = pathParent;

    while (currentParent != null) {
      ResultNode.Builder parentNodeBuilder = ResultNode.builder();

      // Preserve existing children if the parent node is already in the graph.
      if (resultGraph.containsKey(currentParent)) {
        ResultNode existingNode = resultGraph.get(currentParent);
        parentNodeBuilder
            .addChildren(existingNode.getChildren())
            .setTarget(existingNode.isTarget());
      }

      // Add the edge from parent to child.
      boolean isIndirect =
          currentParent.equals(ModuleKey.ROOT) && !rootDirectChildren.contains(currentChild);
      parentNodeBuilder.addChild(
          currentChild, IsExpanded.TRUE, isIndirect ? IsIndirect.TRUE : IsIndirect.FALSE);

      resultGraph.put(currentParent, parentNodeBuilder.build());

      // Move up the path.
      currentChild = currentParent;
      currentParent = bfsParentMap.get(currentChild);
    }
  }

  /**
   * Finds paths from a set of modules to a set of target modules and returns a dependency graph
   * containing these paths.
   *
   * <p>This function performs a breadth-first search (BFS) starting from the {@code from} modules
   * to find paths to the {@code targets}. When a path is found, it's added to the result graph. The
   * search can be configured to stop after finding a single path to each target or to find all
   * possible paths. The final graph is then pruned to the depth specified in the options by {@link
   * ResultGraphPruner}.
   *
   * @param from The set of modules to start the search from.
   * @param targets The set of target modules to find paths to.
   * @param findSinglePath If true, the search for paths to a specific target will stop once the
   *     first path is found.
   * @return An immutable map representing the pruned dependency graph containing the paths.
   */
  ImmutableMap<ModuleKey, ResultNode> expandPathsToTargets(
      ImmutableSet<ModuleKey> from, ImmutableSet<ModuleKey> targets, boolean findSinglePath) {
    // 1. Perform a BFS to find paths from the "from" modules to the "targets".
    // This map tracks the parent of each visited module to reconstruct paths later.
    Map<ModuleKey, ModuleKey> bfsParentMap = new HashMap<>();
    from.stream()
        .filter(this::filterBuiltin)
        .sorted(ModuleKey.LEXICOGRAPHIC_COMPARATOR)
        .forEach(moduleKey -> bfsParentMap.put(moduleKey, ModuleKey.ROOT));
    bfsParentMap.put(ModuleKey.ROOT, null); // The root has no parent.

    Map<ModuleKey, ResultNode> resultGraph = new HashMap<>();
    Deque<ModuleKey> queue = new ArrayDeque<>(from);
    Set<ModuleKey> foundTargets = new HashSet<>();

    while (!queue.isEmpty()) {
      // If we only need one path to each target, and we've found them all, we can stop.
      if (findSinglePath && foundTargets.containsAll(targets)) {
        break;
      }

      ModuleKey currentModuleKey = queue.pop();
      AugmentedModule module = depGraph.get(currentModuleKey);
      ImmutableSortedSet<ModuleKey> dependencies =
          module.getAllDeps(options.includeUnused).keySet().stream()
              .filter(this::filterBuiltin)
              .collect(toImmutableSortedSet(ModuleKey.LEXICOGRAPHIC_COMPARATOR));

      for (ModuleKey depKey : dependencies) {
        // A path to a target is found.
        if (targets.contains(depKey) && !(findSinglePath && foundTargets.contains(depKey))) {
          addPathToResultGraph(resultGraph, bfsParentMap, currentModuleKey, depKey);
          foundTargets.add(depKey);
        }
        // If this dependency hasn't been visited, add it to the queue for traversal.
        if (!bfsParentMap.containsKey(depKey)) {
          bfsParentMap.put(depKey, currentModuleKey);
          queue.add(depKey);
        }
      }
    }

    // 2. Prune the resulting graph containing the found paths to the specified depth.
    return new ResultGraphPruner(MaybeCompleteSet.copyOf(targets), ImmutableMap.copyOf(resultGraph))
        .pruneByDepth();
  }

  /**
   * Expands the full dependency graph starting from a given set of modules and then prunes it to
   * the depth specified in the options.
   *
   * <p>This function first performs a breadth-first traversal to build a complete graph of all
   * dependencies reachable from the {@code from} modules. The {@code from} modules themselves are
   * "pinned" as direct children of the root node in the resulting graph. Finally, it uses {@link
   * ResultGraphPruner} to trim the graph to the requested depth.
   */
  @VisibleForTesting
  ImmutableMap<ModuleKey, ResultNode> expandAndPrune(ImmutableSet<ModuleKey> from) {
    // This map will store the fully expanded dependency graph as ResultNode objects.
    ImmutableMap.Builder<ModuleKey, ResultNode> fullGraphBuilder = new ImmutableMap.Builder<>();

    // 1. Initialize the graph with the ROOT module and its immediate "pinned" children.
    // "Pinned" children are the modules that are explicitly requested to start the graph from.
    ResultNode.Builder rootBuilder = ResultNode.builder();
    ImmutableSet<ModuleKey> rootDirectChildren =
        depGraph.get(ModuleKey.ROOT).getAllDeps(options.includeUnused).keySet();
    ImmutableSortedSet<ModuleKey> pinnedChildren =
        getPinnedChildrenOfRootInTheResultGraph(rootDirectChildren, from).stream()
            .filter(this::filterBuiltin)
            .collect(toImmutableSortedSet(ModuleKey.LEXICOGRAPHIC_COMPARATOR));

    for (ModuleKey pinnedChild : pinnedChildren) {
      boolean isDirect = rootDirectChildren.contains(pinnedChild);
      rootBuilder.addChild(
          pinnedChild, IsExpanded.TRUE, isDirect ? IsIndirect.FALSE : IsIndirect.TRUE);
    }
    fullGraphBuilder.put(ModuleKey.ROOT, rootBuilder.build());

    // 2. Traverse the dependency graph starting from the pinned children (BFS).
    Set<ModuleKey> visited = new HashSet<>(pinnedChildren);
    Deque<ModuleKey> queue = new ArrayDeque<>(pinnedChildren);
    visited.add(ModuleKey.ROOT);

    while (!queue.isEmpty()) {
      ModuleKey currentModuleKey = queue.pop();
      AugmentedModule module = depGraph.get(currentModuleKey);
      ResultNode.Builder nodeBuilder = ResultNode.builder();

      ImmutableSortedSet<ModuleKey> dependencies =
          module.getAllDeps(options.includeUnused).keySet().stream()
              .filter(this::filterBuiltin)
              .collect(toImmutableSortedSet(ModuleKey.LEXICOGRAPHIC_COMPARATOR));

      for (ModuleKey depKey : dependencies) {
        if (visited.contains(depKey)) {
          // This dependency has been seen before, but we add a non-expanded edge to it.
          nodeBuilder.addChild(depKey, IsExpanded.FALSE, IsIndirect.FALSE);
        } else {
          // New dependency found, add it to the queue to visit and mark as expanded.
          nodeBuilder.addChild(depKey, IsExpanded.TRUE, IsIndirect.FALSE);
          visited.add(depKey);
          queue.add(depKey);
        }
      }
      fullGraphBuilder.put(currentModuleKey, nodeBuilder.build());
    }

    // 3. Prune the fully expanded graph based on the specified depth.
    return new ResultGraphPruner(MaybeCompleteSet.completeSet(), fullGraphBuilder.buildOrThrow())
        .pruneByDepth();
  }

  private class ResultGraphPruner {

    private final Map<ModuleKey, ResultNode> oldResult;
    private final Map<ModuleKey, ResultNode.Builder> resultBuilder;
    private final Set<ModuleKey> parentStack;
    private final MaybeCompleteSet<ModuleKey> targets;

    /**
     * Constructs a ResultGraphPruner to prune the result graph after the specified depth.
     *
     * @param targets If not complete, it means that the result graph contains paths to some
     *     specific targets. This will cause some branches to contain, after the specified depths,
     *     some targets or target parents. As any other nodes omitted, transitive edges (embedding
     *     multiple edges) will be stored as <i>indirect</i>.
     * @param oldResult The unpruned result graph.
     */
    ResultGraphPruner(MaybeCompleteSet<ModuleKey> targets, Map<ModuleKey, ResultNode> oldResult) {
      this.oldResult = oldResult;
      this.resultBuilder = new HashMap<>();
      this.parentStack = new HashSet<>();
      this.targets = targets;
    }

    /**
     * Prunes the result tree after the specified depth using DFS (because some nodes may still
     * appear after the max depth).
     */
    private ImmutableMap<ModuleKey, ResultNode> pruneByDepth() {
      if (oldResult.isEmpty()) {
        return ImmutableMap.of();
      }

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
      ImmutableMap<ModuleKey, ResultNode> result =
          resultBuilder.entrySet().stream()
              .collect(
                  toImmutableSortedMap(
                      ModuleKey.LEXICOGRAPHIC_COMPARATOR,
                      Entry::getKey,
                      e -> e.getValue().build()));
      // Filter imports for nodes that were pruned during this process.
      extensionRepoImports = computeRepoImportsTable(result.keySet());
      return result;
    }

    // Handles graph traversal within the specified depth.
    private void visitVisible(
        ModuleKey moduleKey, int depth, ModuleKey parentKey, IsExpanded expanded) {
      parentStack.add(moduleKey);
      ResultNode oldNode = oldResult.get(moduleKey);
      ResultNode.Builder nodeBuilder =
          resultBuilder.computeIfAbsent(moduleKey, k -> ResultNode.builder());

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
          } else if (!targets.isComplete()) {
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

      if (oldNode.isTarget() || isTargetParent(oldNode)) {
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

    private boolean isTargetParent(ResultNode node) {
      return node.getChildren().keys().stream()
          .filter(Predicate.not(parentStack::contains))
          .anyMatch(targets::contains);
    }
  }

  /**
   * Return a sorted list of modules that will be the direct children of the root in the result
   * graph (original root's direct dependencies along with the specified targets).
   */
  private ImmutableSortedSet<ModuleKey> getPinnedChildrenOfRootInTheResultGraph(
      ImmutableSet<ModuleKey> rootDirectDeps, ImmutableSet<ModuleKey> fromTargets) {
    Set<ModuleKey> targetKeys = new HashSet<>(fromTargets);
    if (fromTargets.contains(ModuleKey.ROOT)) {
      targetKeys.remove(ModuleKey.ROOT);
      targetKeys.addAll(rootDirectDeps);
    }
    return ImmutableSortedSet.copyOf(ModuleKey.LEXICOGRAPHIC_COMPARATOR, targetKeys);
  }

  private static boolean intersect(
      MaybeCompleteSet<ModuleExtensionId> a, Set<ModuleExtensionId> b) {
    if (a.isComplete()) {
      return !b.isEmpty();
    }
    return !Collections.disjoint(a.getElementsIfNotComplete(), b);
  }

  /**
   * If the extensionFilter option is set, computes the set of target modules that use the specified
   * extension(s)
   */
  private ImmutableSet<ModuleKey> computeExtensionFilterTargets() {
    if (extensionFilter.isEmpty()) {
      return ImmutableSet.of();
    }
    return depGraph.keySet().stream()
        .filter(this::filterUnused)
        .filter(this::filterBuiltin)
        .filter(k -> intersect(extensionFilter.get(), extensionUsages.column(k).keySet()))
        .collect(toImmutableSet());
  }

  /** Compute the multimap of repo imports to modules for each extension. */
  private ImmutableMap<ModuleExtensionId, ImmutableSetMultimap<String, ModuleKey>>
      computeRepoImportsTable(ImmutableSet<ModuleKey> presentModules) {
    ImmutableMap.Builder<ModuleExtensionId, ImmutableSetMultimap<String, ModuleKey>> resultBuilder =
        new ImmutableMap.Builder<>();
    for (ModuleExtensionId extension : extensionUsages.rowKeySet()) {
      if (extensionFilter.isPresent() && !extensionFilter.get().contains(extension)) {
        continue;
      }
      ImmutableSetMultimap.Builder<ModuleKey, String> modulesToImportsBuilder =
          new ImmutableSetMultimap.Builder<>();
      for (Entry<ModuleKey, ModuleExtensionUsage> usage :
          extensionUsages.rowMap().get(extension).entrySet()) {
        if (!presentModules.contains(usage.getKey())) {
          continue;
        }
        for (ModuleExtensionUsage.Proxy proxy : usage.getValue().getProxies()) {
          modulesToImportsBuilder.putAll(usage.getKey(), proxy.getImports().values());
        }
      }
      resultBuilder.put(extension, modulesToImportsBuilder.build().inverse());
    }
    return resultBuilder.buildOrThrow();
  }

  private boolean filterUnused(ModuleKey key) {
    AugmentedModule module = depGraph.get(key);
    return options.includeUnused || module.isUsed();
  }

  private boolean filterBuiltin(ModuleKey key) {
    return options.includeBuiltin || !isBuiltin(key);
  }

  private String tagToFunctionArgs(AttributeValues attributes) {
    return attributes.attributes().entrySet().stream()
        // show 'name' first for readability, similar to buildifier
        .sorted(Map.Entry.comparingByKey(Comparator.comparing(s -> s.equals("name") ? "" : s)))
        .map(e -> String.format("%s=%s", e.getKey(), Starlark.repr(e.getValue())))
        .collect(joining(", "));
  }

  /** Helper to display show_extension info. */
  private void displayExtension(ModuleExtensionId extension, ImmutableSet<ModuleKey> fromUsages)
      throws InvalidArgumentException {
    printer.printf("## %s:\n", extension.toString());
    printer.println();
    printer.println("Fetched repositories:");
    if (!extensionRepoImports.containsKey(extension)) {
      throw new InvalidArgumentException(
          String.format("No extension %s exists in the dependency graph", extension));
    }
    ImmutableSortedSet<String> usedRepos =
        ImmutableSortedSet.copyOf(extensionRepoImports.get(extension).keySet());
    ImmutableSortedSet<String> unusedRepos =
        ImmutableSortedSet.copyOf(Sets.difference(extensionRepos.get(extension), usedRepos));
    for (String repo : usedRepos) {
      printer.printf(
          "  - %s (imported by %s)\n",
          repo,
          extensionRepoImports.get(extension).get(repo).stream()
              .sorted(ModuleKey.LEXICOGRAPHIC_COMPARATOR)
              .map(ModuleKey::toString)
              .collect(joining(", ")));
    }
    for (String repo : unusedRepos) {
      printer.printf("  - %s\n", repo);
    }
    printer.println();
    if (fromUsages.isEmpty()) {
      fromUsages = ImmutableSet.copyOf(extensionUsages.rowMap().get(extension).keySet());
    }
    for (ModuleKey module : fromUsages) {
      if (!extensionUsages.contains(extension, module)) {
        continue;
      }
      ModuleExtensionUsage usage = extensionUsages.get(extension, module);
      // TODO: maybe consider printing each proxy separately? Might be relevant for included
      //  segments.
      printer.printf(
          "## Usage in %s from %s:%s\n",
          module,
          usage.getProxies().getFirst().getLocation().file(),
          usage.getProxies().getFirst().getLocation().line());

      if (extension.isInnate()) {
        // This is for the special case of "innate" extensions: fake module extensions created by
        // use_repo_rule(). The name of the extension is of the form "<bzl_file_label> <rule_name>".
        // Rule names cannot contain spaces, so we can split on the last space.
        int lastSpace = extension.extensionName().lastIndexOf(' ');
        String rawLabel = extension.extensionName().substring(0, lastSpace);
        String ruleName = extension.extensionName().substring(lastSpace + 1);

        printer.printf("%s = use_repo_rule(\"%s\", \"%s\")\n", ruleName, rawLabel, ruleName);

        for (Tag tag : usage.getTags()) {
          // use_repo_rule creates a fake repo extension with a single tag 'repo'.
          // However, code defensively and print the tag name if it's not 'repo'.
          String callee = ruleName;
          if (!tag.getTagName().equals("repo")) {
            callee = String.format("%s.%s", ruleName, tag.getTagName());
          }
          printer.printf("%s(%s)\n", callee, tagToFunctionArgs(tag.getAttributeValues()));
        }

        // Skip the use_repo part since every call to the repo rule creates a repo that is imported.
        printer.println();

      } else {
        for (Tag tag : usage.getTags()) {
          printer.printf(
              "%s.%s(%s)\n",
              extension.extensionName(),
              tag.getTagName(),
              tagToFunctionArgs(tag.getAttributeValues()));
        }
        printer.printf("use_repo(\n");
        printer.printf("  %s,\n", extension.extensionName());
        for (ModuleExtensionUsage.Proxy proxy : usage.getProxies()) {
          for (Entry<String, String> repo : proxy.getImports().entrySet()) {
            printer.printf(
                "  %s,\n",
                repo.getKey().equals(repo.getValue())
                    ? String.format("\"%s\"", repo.getKey())
                    : String.format("%s=\"%s\"", repo.getKey(), repo.getValue()));
          }
        }
        printer.printf(")\n\n");
      }
    }
  }

  private boolean isBuiltin(ModuleKey key) {
    return key.equals(new ModuleKey("bazel_tools", Version.EMPTY));
  }

  /** A node representing a module that forms the result graph. */
  @AutoValue
  public abstract static class ResultNode {

    /** Whether the module is one of the targets in a paths query. */
    abstract boolean isTarget();

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

    /**
     * Detailed edge type for the {@link ResultNode} graph.
     *
     * @param isExpanded Whether the node should be expanded from this edge (the same node can
     *     appear in multiple places in a flattened graph).
     * @param isIndirect Whether the edge is a direct edge or an indirect (transitive) one.
     * @param isCycle Whether the edge is cycling back inside the flattened graph.
     */
    public record NodeMetadata(IsExpanded isExpanded, IsIndirect isIndirect, IsCycle isCycle) {
      public NodeMetadata {
        requireNonNull(isExpanded, "isExpanded");
        requireNonNull(isIndirect, "isIndirect");
        requireNonNull(isCycle, "isCycle");
      }

      private static NodeMetadata create(
          IsExpanded isExpanded, IsIndirect isIndirect, IsCycle isCycle) {
        return new NodeMetadata(isExpanded, isIndirect, isCycle);
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
      return new AutoValue_ModExecutor_ResultNode.Builder().setTarget(false);
    }

    @AutoValue.Builder
    abstract static class Builder {

      abstract ResultNode.Builder setTarget(boolean value);

      abstract ImmutableSetMultimap.Builder<ModuleKey, NodeMetadata> childrenBuilder();

      @CanIgnoreReturnValue
      final Builder addChild(ModuleKey value, IsExpanded expanded, IsIndirect indirect) {
        childrenBuilder().put(value, NodeMetadata.create(expanded, indirect, IsCycle.FALSE));
        return this;
      }

      @CanIgnoreReturnValue
      final Builder addChildren(ImmutableSetMultimap<ModuleKey, NodeMetadata> children) {
        childrenBuilder().putAll(children);
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

  private void printRepoDefinition(RepoDefinition repoDefinition) {
    RepoRule repoRule = repoDefinition.repoRule();
    printer
        .append("load(\"")
        .append(repoRule.id().bzlFileLabel().getUnambiguousCanonicalForm())
        .append("\", \"")
        .append(repoRule.id().ruleName())
        .append("\")\n");
    printer.append(repoRule.id().ruleName()).append("(\n");
    printer.append("  name = \"").append(repoDefinition.name()).append("\",\n");
    for (Map.Entry<String, Object> attr : repoDefinition.attrValues().attributes().entrySet()) {
      printer
          .append("  ")
          .append(attr.getKey())
          .append(" = ")
          .append(Starlark.repr(attr.getValue()))
          .append(",\n");
    }
    printer.append(")\n");
    // TODO: record and print the call stack for the repo definition itself?
    printer.append("\n");
  }
}
