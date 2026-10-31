// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.query.hash;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Consumer;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * Computes the transitive hashes printed by {@code query --generate_hashes}.
 *
 * <p>The direct hashes of the transitive closure are obtained from Skyframe via {@link
 * DirectTargetHashFunction}, so they are memoized across invocations. They are then combined
 * bottom-up along the dependency graph. For a target that is not part of a dependency cycle, {@code
 * hash(t) = H(direct(t), [(label(d), hash(d)) for d in deps(t)])}. The members of a cycle (a
 * strongly connected component) share a hash of all their direct hashes and edges, salted with each
 * member's label.
 *
 * <p>A {@code toolchain_type} additionally depends on every toolchain registered for it (see {@link
 * RegisteredToolchainsByTypeFunction}), because toolchain resolution needs a configuration. This
 * over-approximates: a change to a toolchain for any platform changes every target that may use its
 * type.
 */
public final class TransitiveHashes {

  /**
   * The hashes of the requested targets, and the errors that make hashes missing or unreliable.
   *
   * @param hashes the hashes of the requested targets that could be hashed
   * @param errors errors of requested targets that could not be hashed, and of their transitive
   *     dependencies whose hashes may be wrong. A dependency that does not exist is not an error:
   *     it is hashed as missing, so creating it changes the hashes of its dependents.
   */
  public record Result(ImmutableMap<Label, String> hashes, ImmutableMap<Label, ErrorInfo> errors) {}

  /** Evaluates Skyframe keys with keep-going semantics. */
  @FunctionalInterface
  public interface Evaluator {
    EvaluationResult<SkyValue> evaluate(ImmutableSet<SkyKey> keys) throws InterruptedException;
  }

  private TransitiveHashes() {}

  /**
   * Computes the transitive hashes of {@code roots}.
   *
   * @param evaluator evaluates the given keys in Skyframe, with keep-going semantics
   * @param progress receives a human-readable progress message after each level of the graph
   */
  public static Result compute(
      Collection<Label> roots, Evaluator evaluator, Consumer<String> progress)
      throws InterruptedException {
    Map<Label, DirectTargetHashFunction.Value> graph = new HashMap<>();
    Map<Label, ErrorInfo> errors = new TreeMap<>();
    ImmutableSet<Label> rootSet = ImmutableSet.copyOf(roots);
    Set<Label> visited = new HashSet<>(roots);
    List<Label> frontier = new ArrayList<>(rootSet);
    // Breadth-first so that each Skyframe evaluation covers a whole level of the graph.
    int level = 0;
    RegisteredToolchains registeredToolchains = new RegisteredToolchains(evaluator);
    while (!frontier.isEmpty()) {
      ImmutableSet<SkyKey> keys =
          frontier.stream().map(DirectTargetHashFunction::key).collect(toImmutableSet());
      EvaluationResult<SkyValue> result = evaluator.evaluate(keys);
      List<Label> next = new ArrayList<>();
      for (Label label : frontier) {
        SkyKey key = DirectTargetHashFunction.key(label);
        DirectTargetHashFunction.Value value = (DirectTargetHashFunction.Value) result.get(key);
        if (value == null) {
          // Deps that do not exist are hashed as missing leaves; see hashOf.
          ErrorInfo error = result.getError(key);
          if (error != null
              && (rootSet.contains(label)
                  || !(error.getException() instanceof NoSuchThingException))) {
            errors.put(label, error);
          }
          continue;
        }
        if (value.isToolchainType()) {
          value = registeredToolchains.addToolchains(label, value, errors);
        }
        graph.put(label, value);
        for (Label dep : value.getDeps()) {
          if (visited.add(dep)) {
            next.add(dep);
          }
        }
      }
      frontier = next;
      progress.accept(
          String.format(
              "Hashing: visited %d targets (depth %d, %d pending)",
              graph.size(), ++level, frontier.size()));
    }

    progress.accept(String.format("Hashing: combining hashes of %d targets", graph.size()));
    Map<Label, byte[]> hashes = combine(graph);
    ImmutableMap.Builder<Label, String> rootHashes = ImmutableMap.builder();
    for (Label root : rootSet) {
      byte[] hash = hashes.get(root);
      if (hash != null) {
        rootHashes.put(root, BaseEncoding.base16().lowerCase().encode(hash));
      }
    }
    return new Result(rootHashes.buildOrThrow(), ImmutableMap.copyOf(errors));
  }

  /** Lazily evaluates {@link RegisteredToolchainsByTypeFunction}, once per computation. */
  private static final class RegisteredToolchains {
    private final Evaluator evaluator;
    private boolean evaluated = false;
    @Nullable private RegisteredToolchainsByTypeFunction.Value value;
    @Nullable private ErrorInfo error;

    RegisteredToolchains(Evaluator evaluator) {
      this.evaluator = evaluator;
    }

    /** Returns {@code toolchainType}'s value with edges to the toolchains registered for it. */
    DirectTargetHashFunction.Value addToolchains(
        Label toolchainType, DirectTargetHashFunction.Value typeValue, Map<Label, ErrorInfo> errors)
        throws InterruptedException {
      if (!evaluated) {
        EvaluationResult<SkyValue> result =
            evaluator.evaluate(ImmutableSet.of(RegisteredToolchainsByTypeFunction.KEY));
        value =
            (RegisteredToolchainsByTypeFunction.Value)
                result.get(RegisteredToolchainsByTypeFunction.KEY);
        error = result.getError(RegisteredToolchainsByTypeFunction.KEY);
        evaluated = true;
      }
      if (value == null) {
        // The registered toolchains could not be determined (an invalid register_toolchains
        // pattern, for instance). Report it, and hash that fact rather than silently dropping the
        // edges.
        if (error != null) {
          errors.put(toolchainType, error);
        }
        return new DirectTargetHashFunction.Value(
            new Fingerprint()
                .addBytes(typeValue.getDirectHash())
                .addString("<registered toolchains unavailable>")
                .digestAndReset(),
            typeValue.getDeps(),
            /* toolchainType= */ true);
      }
      TreeSet<Label> deps = new TreeSet<>(typeValue.getDeps());
      deps.addAll(value.getToolchains(toolchainType));
      deps.remove(toolchainType);
      return new DirectTargetHashFunction.Value(
          typeValue.getDirectHash(), ImmutableList.copyOf(deps), /* toolchainType= */ true);
    }
  }

  /** Combines direct hashes into transitive hashes using Tarjan's SCC algorithm. */
  static Map<Label, byte[]> combine(Map<Label, DirectTargetHashFunction.Value> graph) {
    Map<Label, Node<Label>> nodes = new HashMap<>(graph.size());
    graph.forEach(
        (label, value) -> nodes.put(label, new Node<>(value.getDirectHash(), value.getDeps())));
    return combine(nodes, Comparator.naturalOrder(), Label::getUnambiguousCanonicalForm);
  }

  /**
   * A node of the graph to hash: its direct hash and its dependencies, in a deterministic order.
   * Dependencies that are not part of the graph are hashed as missing leaves.
   */
  public record Node<K>(byte[] directHash, List<K> deps) {}

  /**
   * Combines the direct hashes of {@code graph} into transitive hashes using Tarjan's SCC
   * algorithm.
   *
   * @param order a deterministic total order of the nodes, used to order the members of a cycle
   * @param name a stable, unique name of each node, hashed as the identity of edges and cycle
   *     members
   */
  public static <K> Map<K, byte[]> combine(
      Map<K, Node<K>> graph, Comparator<? super K> order, Function<? super K, String> name) {
    Map<K, byte[]> hashes = new HashMap<>(graph.size());
    Map<K, Integer> index = new HashMap<>(graph.size());
    Map<K, Integer> lowLink = new HashMap<>(graph.size());
    Deque<K> stack = new ArrayDeque<>();
    Set<K> onStack = new HashSet<>();

    for (K root : graph.keySet()) {
      if (index.containsKey(root)) {
        continue;
      }
      Deque<Frame<K>> work = new ArrayDeque<>();
      push(root, graph, index, lowLink, stack, onStack, work);
      while (!work.isEmpty()) {
        Frame<K> frame = work.peek();
        if (frame.next()[0] < frame.deps().size()) {
          K dep = frame.deps().get(frame.next()[0]++);
          if (!graph.containsKey(dep)) {
            continue; // A missing leaf.
          }
          if (!index.containsKey(dep)) {
            push(dep, graph, index, lowLink, stack, onStack, work);
          } else if (onStack.contains(dep)) {
            lowLink.merge(frame.node(), index.get(dep), Math::min);
          }
          continue;
        }
        work.pop();
        K node = frame.node();
        if (lowLink.get(node).equals(index.get(node))) {
          List<K> component = new ArrayList<>();
          K member;
          do {
            member = stack.pop();
            onStack.remove(member);
            component.add(member);
          } while (!member.equals(node));
          hashComponent(component, graph, order, name, hashes);
        }
        if (!work.isEmpty()) {
          lowLink.merge(work.peek().node(), lowLink.get(node), Math::min);
        }
      }
    }
    return hashes;
  }

  /** An explicit DFS frame, to avoid overflowing the Java stack on deep graphs. */
  private record Frame<K>(K node, List<K> deps, int[] next) {}

  private static <K> void push(
      K node,
      Map<K, Node<K>> graph,
      Map<K, Integer> index,
      Map<K, Integer> lowLink,
      Deque<K> stack,
      Set<K> onStack,
      Deque<Frame<K>> work) {
    int i = index.size();
    index.put(node, i);
    lowLink.put(node, i);
    stack.push(node);
    onStack.add(node);
    work.push(new Frame<>(node, graph.get(node).deps(), new int[] {0}));
  }

  /**
   * Hashes one strongly connected component. All deps outside of it have already been hashed,
   * because Tarjan's algorithm emits components in reverse topological order.
   */
  private static <K> void hashComponent(
      List<K> component,
      Map<K, Node<K>> graph,
      Comparator<? super K> order,
      Function<? super K, String> name,
      Map<K, byte[]> hashes) {
    if (component.size() == 1 && !graph.get(component.get(0)).deps().contains(component.get(0))) {
      // Not part of a cycle.
      K node = component.get(0);
      Fingerprint fp = new Fingerprint();
      fp.addBytes(graph.get(node).directHash());
      addDeps(fp, graph.get(node), /* component= */ Set.of(), name, hashes);
      hashes.put(node, fp.digestAndReset());
      return;
    }
    List<K> members = new ArrayList<>(component);
    members.sort(order);
    Set<K> memberSet = new HashSet<>(members);
    Fingerprint fp = new Fingerprint();
    fp.addString("<cycle>");
    for (K member : members) {
      fp.addString(name.apply(member));
      fp.addBytes(graph.get(member).directHash());
      addDeps(fp, graph.get(member), memberSet, name, hashes);
    }
    byte[] componentHash = fp.digestAndReset();
    for (K member : members) {
      hashes.put(
          member,
          new Fingerprint().addBytes(componentHash).addString(name.apply(member)).digestAndReset());
    }
  }

  private static <K> void addDeps(
      Fingerprint fp,
      Node<K> node,
      Set<K> component,
      Function<? super K, String> name,
      Map<K, byte[]> hashes) {
    for (K dep : node.deps()) {
      fp.addString(name.apply(dep));
      if (component.contains(dep)) {
        continue; // Edges within a cycle are covered by the component hash.
      }
      byte[] depHash = hashes.get(dep);
      // A dep that could not be loaded contributes deterministically, so that it turning into an
      // existing target is detected.
      fp.addString(depHash == null ? "<missing>" : "");
      if (depHash != null) {
        fp.addBytes(depHash);
      }
    }
  }
}
