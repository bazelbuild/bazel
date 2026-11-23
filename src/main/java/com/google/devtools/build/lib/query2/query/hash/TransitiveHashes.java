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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.compacthashmap.CompactHashMap;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.protobuf.ByteString;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
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
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.ToLongFunction;
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
  private static final int MIN_PARALLEL_FRONTIER_KEYS = 32_768;
  private static final int MIN_ROOTS_FOR_BATCHING = 1024;
  private static final int MIN_ROOTS_PER_BATCHED_PACKAGE = 8;
  private static final int MAX_ROOTS_PER_BATCHED_PACKAGE = 1024;

  /**
   * The hashes of the requested targets, and the errors that make hashes missing or unreliable.
   *
   * @param hashes the hashes of the requested targets that could be hashed
   * @param errors errors of requested targets that could not be hashed, and of their transitive
   *     dependencies whose hashes may be wrong. A dependency that does not exist is not an error:
   *     it is hashed as missing, so creating it changes the hashes of its dependents.
   * @param timing elapsed time for the computation phases
   */
  public record Result(
      ImmutableMap<Label, ByteString> hashes,
      ImmutableMap<Label, ErrorInfo> errors,
      ImmutableMap<String, FailureImpact> failureImpact,
      int unlistedFailureImpact,
      Timing timing) {}

  /** Wall time by phase, in milliseconds. */
  public record Timing(long discoveryMs, long combineMs, long assembleMs) {}

  /** Requested targets reaching a direct failure, and those reaching only that failure. */
  public record FailureImpact(int affected, int exclusive) {}

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
    return compute(roots, evaluator, progress, false);
  }

  /** Computes hashes and, if requested, counts requested roots affected by each failure group. */
  public static Result compute(
      Collection<Label> roots,
      Evaluator evaluator,
      Consumer<String> progress,
      boolean reportFailureImpact)
      throws InterruptedException {
    return compute(
        roots, evaluator, progress, reportFailureImpact, /* batchDirectRuleHashes= */ false);
  }

  /** Computes hashes, optionally grouping direct rule hashes in large root sets by package. */
  public static Result compute(
      Collection<Label> roots,
      Evaluator evaluator,
      Consumer<String> progress,
      boolean reportFailureImpact,
      boolean batchDirectRuleHashes)
      throws InterruptedException {
    long discoveryStart = System.nanoTime();
    Map<Label, State<Label>> graph = CompactHashMap.createWithExpectedSize(roots.size());
    Set<Label> failed = new HashSet<>();
    List<Label> absent = new ArrayList<>();
    Map<Label, ErrorInfo> errors = new TreeMap<>();
    ImmutableSet<Label> rootSet = ImmutableSet.copyOf(roots);
    // Keep scheduled labels in the graph map so every label is evaluated at most once.
    State<Label> pending = new State<>(null, null, null, false);
    rootSet.forEach(label -> graph.put(label, pending));
    int resolved = 0;
    List<Label> frontier = rootSet.asList();
    // Breadth-first so that each Skyframe evaluation covers a whole level of the graph.
    int level = 0;
    RegisteredToolchains registeredToolchains = new RegisteredToolchains(evaluator);
    Map<PackageIdentifier, DirectTargetHashFunction.PackageHashes> packageHashes =
        batchDirectRuleHashes && rootSet.size() >= MIN_ROOTS_FOR_BATCHING
            ? evaluateRootPackages(rootSet, evaluator)
            : Map.of();
    while (!frontier.isEmpty()) {
      ImmutableSet<SkyKey> keys = buildFrontierKeys(frontier, packageHashes);
      var result =
          keys.isEmpty() ? EvaluationResult.<SkyValue>builder().build() : evaluator.evaluate(keys);
      List<Label> next = new ArrayList<>();
      var keyIterator = keys.iterator();
      for (Label label : frontier) {
        DirectTargetHashFunction.Value value = getPackageHash(packageHashes, label);
        if (value == null) {
          SkyKey key = keyIterator.next();
          value = (DirectTargetHashFunction.Value) result.get(key);
          if (value == null) {
            ErrorInfo error = result.getError(key);
            // A missing dependency is hashed as absent only when Skyframe reports it as such.
            // Fetch failures, I/O errors, and partially loaded packages may still contain the label.
            if (rootSet.contains(label) || !isKnownAbsent(error)) {
              failed.add(label);
              if (error != null) {
                errors.put(label, error);
              }
            } else {
              absent.add(label);
            }
            continue;
          }
        }
        if (value.isToolchainType()) {
          value = registeredToolchains.addToolchains(label, value, errors);
          if (value == null) {
            failed.add(label);
            continue;
          }
        }
        graph.put(label, new State<>(label, value.getDirectHash(), value.getDeps(), false));
        resolved++;
        for (Label dep : value.getDeps()) {
          if (graph.putIfAbsent(dep, pending) == null) {
            next.add(dep);
          }
        }
      }
      packageHashes = Map.of();
      frontier = next;
      progress.accept(
          String.format(
              "Hashing: visited %d targets (depth %d, %d pending)",
              resolved, ++level, frontier.size()));
    }

    failed.forEach(graph::remove);
    absent.forEach(graph::remove);
    progress.accept(String.format("Hashing: combining hashes of %d targets", resolved));
    long combineStart = System.nanoTime();
    var failureImpact =
        reportFailureImpact && !failed.isEmpty() ? new FailureImpactCollector(failed) : null;
    ToLongFunction<Label> failedMask =
        failureImpact == null ? label -> failed.contains(label) ? 1L : 0L : failureImpact::mask;
    combineStates(graph, Comparator.naturalOrder(), Label::getUnambiguousCanonicalForm, failedMask);
    long assembleStart = System.nanoTime();
    ImmutableMap.Builder<Label, ByteString> rootHashes =
        ImmutableMap.builderWithExpectedSize(roots.size());
    for (Label root : rootSet) {
      State<Label> state = graph.get(root);
      if (state != null && state.hash != null) {
        rootHashes.put(root, ByteString.copyFrom(state.hash));
      } else if (failureImpact != null) {
        failureImpact.record(state == null ? failureImpact.mask(root) : state.failures);
      }
    }
    var hashes = rootHashes.buildOrThrow();
    var sortedErrors = ImmutableMap.copyOf(errors);
    ImmutableMap<String, FailureImpact> impact =
        failureImpact == null ? ImmutableMap.of() : failureImpact.build();
    int unlisted = failureImpact == null ? 0 : failureImpact.unlisted();
    long end = System.nanoTime();
    var profiler = Profiler.instance();
    profiler.logSimpleTask(combineStart, assembleStart, ProfilerTask.INFO, "NativeQuery.combine");
    profiler.logSimpleTask(assembleStart, end, ProfilerTask.INFO, "NativeQuery.assemble");
    return new Result(
        hashes,
        sortedErrors,
        impact,
        unlisted,
        new Timing(
            TimeUnit.NANOSECONDS.toMillis(combineStart - discoveryStart),
            TimeUnit.NANOSECONDS.toMillis(assembleStart - combineStart),
            TimeUnit.NANOSECONDS.toMillis(end - assembleStart)));
  }

  private static ImmutableSet<SkyKey> buildFrontierKeys(
      List<Label> frontier, Map<PackageIdentifier, DirectTargetHashFunction.PackageHashes> packages)
      throws InterruptedException {
    if (packages.isEmpty()
        && frontier.size() >= MIN_PARALLEL_FRONTIER_KEYS
        && ForkJoinPool.getCommonPoolParallelism() > 1) {
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }
      SkyKey[] keys = new SkyKey[frontier.size()];
      Arrays.parallelSetAll(keys, index -> DirectTargetHashFunction.key(frontier.get(index)));
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }
      return ImmutableSet.<SkyKey>builderWithExpectedSize(keys.length).add(keys).build();
    }
    ImmutableSet.Builder<SkyKey> keys = ImmutableSet.builderWithExpectedSize(frontier.size());
    for (Label label : frontier) {
      if (getPackageHash(packages, label) == null) {
        keys.add(DirectTargetHashFunction.key(label));
      }
    }
    return keys.build();
  }

  private static Map<PackageIdentifier, DirectTargetHashFunction.PackageHashes>
      evaluateRootPackages(ImmutableSet<Label> roots, Evaluator evaluator)
          throws InterruptedException {
    Map<PackageIdentifier, Integer> counts = new HashMap<>();
    for (Label label : roots) {
      if (label.getRepository().isMain() && !label.getName().contains("/")) {
        counts.merge(label.getPackageIdentifier(), 1, Integer::sum);
      }
    }
    var builder = ImmutableSet.<SkyKey>builder();
    for (var entry : counts.entrySet()) {
      if (entry.getValue() >= MIN_ROOTS_PER_BATCHED_PACKAGE
          && entry.getValue() <= MAX_ROOTS_PER_BATCHED_PACKAGE) {
        builder.add(DirectTargetHashFunction.packageKey(entry.getKey()));
      }
    }
    var keys = builder.build();
    if (keys.isEmpty()) {
      return Map.of();
    }
    var values = evaluator.evaluate(keys);
    Map<PackageIdentifier, DirectTargetHashFunction.PackageHashes> packages = new HashMap<>();
    for (SkyKey key : keys) {
      if (values.get(key) instanceof DirectTargetHashFunction.PackageHashes hashes
          && !hashes.rules().isEmpty()) {
        packages.put((PackageIdentifier) key.argument(), hashes);
      }
    }
    return packages;
  }

  @Nullable
  private static DirectTargetHashFunction.Value getPackageHash(
      Map<PackageIdentifier, DirectTargetHashFunction.PackageHashes> packages, Label label) {
    if (packages.isEmpty()) {
      return null;
    }
    var hashes = packages.get(label.getPackageIdentifier());
    return hashes == null ? null : hashes.rules().get(label.getName());
  }

  private static final class FailureImpactCollector {
    private final List<String> reported;
    private final Map<Label, Long> masks;
    private final int[] affected = new int[Long.SIZE];
    private final int[] exclusive;

    FailureImpactCollector(Set<Label> failed) {
      // The last bit retains unlisted failures so they cannot make a listed group look exclusive.
      Set<String> groups = new TreeSet<>();
      failed.forEach(label -> groups.add(failureGroup(label)));
      this.reported = groups.stream().limit(Long.SIZE - 1).toList();
      this.exclusive = new int[reported.size()];
      Map<String, Long> groupMasks = HashMap.newHashMap(reported.size());
      for (int i = 0; i < reported.size(); i++) {
        groupMasks.put(reported.get(i), 1L << i);
      }
      this.masks = HashMap.newHashMap(failed.size());
      failed.forEach(
          label -> masks.put(label, groupMasks.getOrDefault(failureGroup(label), Long.MIN_VALUE)));
    }

    long mask(Label label) {
      return masks.getOrDefault(label, 0L);
    }

    void record(long reachable) {
      int only = Long.numberOfTrailingZeros(reachable);
      if (only < exclusive.length && Long.bitCount(reachable) == 1) {
        exclusive[only]++;
      }
      while (reachable != 0) {
        affected[Long.numberOfTrailingZeros(reachable)]++;
        reachable &= reachable - 1;
      }
    }

    ImmutableMap<String, FailureImpact> build() {
      var impact = ImmutableMap.<String, FailureImpact>builderWithExpectedSize(reported.size());
      for (int i = 0; i < reported.size(); i++) {
        impact.put(reported.get(i), new FailureImpact(affected[i], exclusive[i]));
      }
      return impact.buildOrThrow();
    }

    int unlisted() {
      return affected[Long.SIZE - 1];
    }

    private static String failureGroup(Label label) {
      var repository = label.getRepository();
      return repository.isMain()
          ? label.getUnambiguousCanonicalForm()
          : repository.getCanonicalForm();
    }
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
    graph.forEach(
        (key, node) ->
            states.put(key, new State<>(key, node.directHash(), node.deps(), failedKeys.contains(key))));
    combineStates(states, order, name, key -> failedKeys.contains(key) ? 1 : 0);
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
      ToLongFunction<? super K> failedMask) {
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
        if (state.nextDep < state.deps.size()) {
          K depKey = state.deps.get(state.nextDep++);
          State<K> dep = states.get(depKey);
          if (dep == null) {
            state.failures |= failedMask.applyAsLong(depKey);
          } else if (dep.index < 0) {
            push(dep, nextIndex++, stack, work);
          } else if (dep.component < 0) {
            state.lowLink = Math.min(state.lowLink, dep.index);
            state.selfEdge |= state == dep;
          } else {
            state.failures |= dep.failures;
          }
          continue;
        }

        work.pop();
        if (state.lowLink == state.index) {
          component.clear();
          State<K> member;
          long failures = 0;
          do {
            member = stack.pop();
            member.component = state.index;
            component.add(member);
            failures |= member.failures;
          } while (member != state);
          if (failures != 0) {
            for (State<K> failedMember : component) {
              failedMember.failures = failures;
            }
          } else {
            hashComponent(component, states, order, name, fp);
          }
        }
        if (!work.isEmpty()) {
          State<K> parent = work.peek();
          parent.lowLink = Math.min(parent.lowLink, state.lowLink);
          parent.failures |= state.failures;
        }
      }
    }

  }

  private static final class State<K> {
    final K key;
    final byte[] directHash;
    final List<K> deps;
    int index = -1;
    int lowLink;
    int component = -1;
    int nextDep;
    boolean selfEdge;
    long failures;
    byte[] hash;
    @Nullable String displayName;

    State(K key, byte[] directHash, List<K> deps, boolean failed) {
      this.key = key;
      this.directHash = directHash;
      this.deps = deps;
      this.failures = failed ? 1 : 0;
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
      fp.addBytes(first.directHash);
      addDeps(fp, first, states, name);
      first.hash = fp.digestAndReset();
      return;
    }
    component.sort((a, b) -> order.compare(a.key, b.key));
    fp.addString("<cycle>");
    for (State<K> member : component) {
      fp.addString(member.displayName(name));
      fp.addBytes(member.directHash);
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
    for (K key : state.deps) {
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
