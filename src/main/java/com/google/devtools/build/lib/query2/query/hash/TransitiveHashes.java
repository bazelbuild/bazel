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
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading;
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
    Map<Label, State<Label>> graph = HashMap.newHashMap(roots.size());
    Set<Label> failed = new HashSet<>();
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
      for (SkyKey key : keys) {
        Label label = (Label) key.argument();
        DirectTargetHashFunction.Value value = (DirectTargetHashFunction.Value) result.get(key);
        if (value == null) {
          ErrorInfo error = result.getError(key);
          // Keep the PR's missing-leaf marker only for a proven absent dependency.
          // Fetch failures, I/O errors, and partially loaded packages may still contain the label.
          if (rootSet.contains(label) || !isKnownAbsent(error)) {
            failed.add(label);
            if (error != null) {
              errors.put(label, error);
            }
          }
          continue;
        }
        if (value.isToolchainType()) {
          value = registeredToolchains.addToolchains(label, value, errors);
          if (value == null) {
            failed.add(label);
            continue;
          }
        }
        graph.put(label, new State<>(label, new Node<>(value.getDirectHash(), value.getDeps()), false));
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
    combineStates(graph, Comparator.naturalOrder(), Label::getUnambiguousCanonicalForm, failed);
    ImmutableMap.Builder<Label, String> rootHashes = ImmutableMap.builderWithExpectedSize(roots.size());
    for (Label root : rootSet) {
      State<Label> state = graph.get(root);
      if (state != null && state.hash != null) {
        rootHashes.put(root, BaseEncoding.base16().lowerCase().encode(state.hash));
      }
    }
    return new Result(rootHashes.buildOrThrow(), ImmutableMap.copyOf(errors));
  }

  private static boolean isKnownAbsent(@Nullable ErrorInfo error) {
    if (error == null || !(error.getException() instanceof BuildFileNotFoundException missing)) {
      return false;
    }
    var detail = missing.getDetailedExitCode().getFailureDetail();
    return detail != null
        && detail.getPackageLoading().getCode() == PackageLoading.Code.BUILD_FILE_MISSING;
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
    @Nullable
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
        if (error != null) {
          errors.put(toolchainType, error);
        }
        return null;
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
    return combine(graph, order, name, Set.of());
  }

  /** Omits any component that includes or transitively depends on a failed key. */
  public static <K> Map<K, byte[]> combine(
      Map<K, Node<K>> graph,
      Comparator<? super K> order,
      Function<? super K, String> name,
      Set<K> failedKeys) {
    Map<K, State<K>> states = HashMap.newHashMap(graph.size());
    graph.forEach((key, node) -> states.put(key, new State<>(key, node, failedKeys.contains(key))));
    combineStates(states, order, name, failedKeys);
    Map<K, byte[]> hashes = HashMap.newHashMap(states.size());
    states.forEach(
        (key, state) -> {
          if (state.hash != null) {
            hashes.put(key, state.hash);
          }
        });
    return hashes;
  }

  private static <K> void combineStates(
      Map<K, State<K>> states,
      Comparator<? super K> order,
      Function<? super K, String> name,
      Set<K> failedKeys) {
    Deque<State<K>> stack = new ArrayDeque<>();
    Deque<State<K>> work = new ArrayDeque<>();
    List<State<K>> component = new ArrayList<>();
    Fingerprint fp = new Fingerprint();
    int nextIndex = 0;

    for (State<K> root : states.values()) {
      if (root.index >= 0) {
        continue;
      }
      push(root, nextIndex++, stack, work);
      while (!work.isEmpty()) {
        State<K> state = work.peek();
        if (state.nextDep < state.node.deps().size()) {
          K depKey = state.node.deps().get(state.nextDep++);
          State<K> dep = states.get(depKey);
          if (dep == null) {
            state.failed |= failedKeys.contains(depKey);
          } else if (dep.index < 0) {
            push(dep, nextIndex++, stack, work);
          } else if (dep.component < 0) {
            state.lowLink = Math.min(state.lowLink, dep.index);
            state.selfEdge |= state == dep;
          } else {
            state.failed |= dep.failed;
          }
          continue;
        }

        work.pop();
        if (state.lowLink == state.index) {
          component.clear();
          State<K> member;
          boolean failed = false;
          do {
            member = stack.pop();
            member.component = state.index;
            component.add(member);
            failed |= member.failed;
          } while (member != state);
          if (failed) {
            for (State<K> failedMember : component) {
              failedMember.failed = true;
            }
          } else {
            hashComponent(component, states, order, name, fp);
          }
        }
        if (!work.isEmpty()) {
          State<K> parent = work.peek();
          parent.lowLink = Math.min(parent.lowLink, state.lowLink);
          parent.failed |= state.failed;
        }
      }
    }

  }

  private static final class State<K> {
    final K key;
    final Node<K> node;
    int index = -1;
    int lowLink;
    int component = -1;
    int nextDep;
    boolean selfEdge;
    boolean failed;
    byte[] hash;
    @Nullable String displayName;

    State(K key, Node<K> node, boolean failed) {
      this.key = key;
      this.node = node;
      this.failed = failed;
    }

    String displayName(Function<? super K, String> name) {
      if (displayName == null) {
        displayName = name.apply(key);
      }
      return displayName;
    }
  }

  private static <K> void push(
      State<K> state, int index, Deque<State<K>> stack, Deque<State<K>> work) {
    state.index = state.lowLink = index;
    stack.push(state);
    work.push(state);
  }

  private static <K> void hashComponent(
      List<State<K>> component,
      Map<K, State<K>> states,
      Comparator<? super K> order,
      Function<? super K, String> name,
      Fingerprint fp) {
    State<K> first = component.getFirst();
    if (component.size() == 1 && !first.selfEdge) {
      fp.addBytes(first.node.directHash());
      addDeps(fp, first, states, name);
      first.hash = fp.digestAndReset();
      return;
    }
    component.sort((a, b) -> order.compare(a.key, b.key));
    fp.addString("<cycle>");
    for (State<K> member : component) {
      fp.addString(member.displayName(name));
      fp.addBytes(member.node.directHash());
      addDeps(fp, member, states, name);
    }
    byte[] componentHash = fp.digestAndReset();
    for (State<K> member : component) {
      member.hash = fp.addBytes(componentHash).addString(member.displayName(name)).digestAndReset();
    }
  }

  private static <K> void addDeps(
      Fingerprint fp,
      State<K> state,
      Map<K, State<K>> states,
      Function<? super K, String> name) {
    for (K key : state.node.deps()) {
      State<K> dep = states.get(key);
      fp.addString(dep == null ? name.apply(key) : dep.displayName(name));
      if (dep != null && dep.component == state.component) {
        continue; // Edges within a cycle are covered by the component hash.
      }
      byte[] depHash = dep == null ? null : dep.hash;
      fp.addString(depHash == null ? "<missing>" : "");
      if (depHash != null) {
        fp.addBytes(depHash);
      }
    }
  }
}
