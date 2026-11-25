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
package com.google.devtools.build.lib.query2.cquery;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata.AnalysisTimeFileContents;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.AnalysisRepoMappingDigest;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.StarlarkAspectClass;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment.TopLevelConfigurations;
import com.google.devtools.build.lib.query2.SkyQueryEnvironment;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.query.hash.DirectTargetHashFunction;
import com.google.devtools.build.lib.server.FailureDetails.Query;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.RuleConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.util.EnvVar;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.BiConsumer;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * cquery output formatter for {@code --generate_hashes}. Prints each configured target in the
 * result followed by a deterministic hash of it and its transitive configured dependencies.
 *
 * <p>Unlike {@code query --generate_hashes}, the graph that is hashed is the configured target
 * graph: {@code select()}s are resolved, toolchains and execution platforms are the ones that were
 * actually chosen, and each node is hashed together with its configuration. A target's own hash is
 * the {@link DirectTargetHashFunction} hash of its unconfigured target (attribute values, rule
 * definition, or source file contents), combined with its build options and the current values of
 * explicitly inherited action and test environment variables.
 */
class HashesOutputFormatterCallback extends CqueryThreadsafeCallback {

  @FunctionalInterface
  interface DepsRetriever {
    @Nullable
    Iterable<CqueryNode> getDirectDeps(
        CqueryNode target,
        BiConsumer<AspectKey, AspectValue> aspectObserver,
        @Nullable Predicate<SkyKey> tryEmitInternedDependency)
        throws InterruptedException;
  }

  static final String NAME = "generate_hashes";

  private final SkyframeExecutor skyframeExecutor;
  private final DepsRetriever depsRetriever;
  // Reuse the walk's availability read. Only retain digests from actual lookups.
  private final Map<AspectKey, AnalysisRepoMappingDigest> aspectMappings = new ConcurrentHashMap<>();
  private final BiConsumer<AspectKey, AspectValue> aspectObserver = this::observeAspect;
  @Nullable private final TopLevelConfigurations topLevelConfigurations;
  private final LabelPrinter labelPrinter;
  private final boolean keepGoing;
  private final ImmutableMap<String, String> clientEnvironment;
  // Scoped to this command. Configured target and action keys remain unchanged.
  private final Map<BuildOptions, EnvironmentChecksums> environmentChecksums =
      new ConcurrentHashMap<>();

  HashesOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<CqueryNode> accessor,
      DepsRetriever depsRetriever,
      @Nullable TopLevelConfigurations topLevelConfigurations,
      LabelPrinter labelPrinter,
      boolean keepGoing,
      ImmutableMap<String, String> clientEnvironment) {
    super(eventHandler, options, out, skyframeExecutor, accessor, /* uniquifyResults= */ false);
    this.skyframeExecutor = skyframeExecutor;
    this.depsRetriever = depsRetriever;
    this.topLevelConfigurations = topLevelConfigurations;
    this.labelPrinter = labelPrinter;
    this.keepGoing = keepGoing;
    this.clientEnvironment = clientEnvironment;
  }

  @Override
  public String getName() {
    return NAME;
  }

  private int failures = 0;

  @FunctionalInterface
  private interface RangeWork {
    void run(int first, int limit) throws IOException, InterruptedException;
  }

  private static void ranges(ExecutorService pool, int first, int limit, int stride, RangeWork work)
      throws IOException, InterruptedException {
    var tasks = new ArrayList<Callable<Void>>();
    for (int begin = first; begin < limit; begin += stride) {
      int left = begin;
      int right = Math.min(limit, begin + stride);
      tasks.add(
          () -> {
            work.run(left, right);
            return null;
          });
    }
    for (var future : pool.invokeAll(tasks)) {
      try {
        future.get();
      } catch (ExecutionException e) {
        var cause = e.getCause();
        if (cause instanceof IOException io) {
          throw io;
        }
        if (cause instanceof InterruptedException interrupted) {
          throw interrupted;
        }
        if (cause instanceof RuntimeException runtime) {
          throw runtime;
        }
        if (cause instanceof Error error) {
          throw error;
        }
        throw new IOException(cause);
      }
    }
  }

  @Override
  public void process(Iterable<CqueryNode> partialResult)
      throws QueryException, InterruptedException {
    super.process(partialResult);
    if (failures > 0) {
      throw new QueryException(
          String.format("failed to compute hashes for %d target(s)", failures),
          Query.Code.BUILD_FILE_ERROR);
    }
  }

  @Override
  public void processOutput(Iterable<CqueryNode> partialResult)
      throws IOException, InterruptedException {
    List<HashTarget> roots = new ArrayList<>();
    Set<HashTarget> failed = new HashSet<>();
    int available = Runtime.getRuntime().availableProcessors();
    int requested = options.getHashWorkers();
    int workers = requested == 0 ? available : Math.min(available, requested);
    boolean parallel = options.useParallelHashing();
    ExecutorService pool = workers > 1 ? Executors.newFixedThreadPool(workers) : null;
    List<HashTarget> nodes;
    try {
      try (var ignored = Profiler.instance().profile("CQHASH walk")) {
        nodes = collectGraph(partialResult, roots, failed, pool);
      }
      try (var ignored = Profiler.instance().profile("CQHASH direct")) {
        if (parallel) {
          computeDistinctDirectHashes(nodes, failed, workers, pool);
        } else {
          for (List<HashTarget> batch : Lists.partition(nodes, 16384)) {
            computeDirectHashes(batch, failed);
          }
        }
      }
    } finally {
      if (pool != null) {
        pool.shutdownNow();
        pool.close();
      }
    }
    failures += failed.size();
    if (!keepGoing && failures > 0) {
      return;
    }
    eventHandler.handle(
        Event.progress(String.format("Hashing: combining hashes of %d targets", nodes.size())));
    try (var ignored = Profiler.instance().profile("CQHASH combine")) {
      if (parallel) {
        CompactConfiguredHashes.combineParallel(nodes, failed, workers);
      } else {
        CompactConfiguredHashes.combine(nodes, failed);
      }
    } catch (UncheckedIOException collision) {
      throw collision.getCause();
    }
    try (var ignored = Profiler.instance().profile("CQHASH emit")) {
      for (HashTarget target : roots) {
        CqueryNode root = target.node;
        byte[] hash = target.hash;
        if (topLevelConfigurations != null) {
          writeTopLevelRow(root, hash);
          continue;
        }
        if (hash == null) {
          continue;
        }
        String description =
            String.format(
                "%s (%s)",
                root.getDescription(labelPrinter),
                shortId(getConfiguration(root.getConfigurationKey())));
        printStream
            .append(description)
            .append(' ')
            .append(BaseEncoding.base16().lowerCase().encode(hash))
            .append(options.getLineTerminator());
      }
      // Keep-going reports individual failed digests as blank rows and still completes the
      // stream. Fail-fast returns before emitting any rows or the completion marker.
      if (options.getHashStreamReceipt()) {
        printStream.append("# complete\n");
      }
      printStream.flush();
    }
  }

  private void writeTopLevelRow(CqueryNode root, @Nullable byte[] hash) throws IOException {
    String checksum = root.getConfigurationChecksum();
    CqueryNode actual = root.getActual();
    printStream
        .append(internalToUnicode(root.getOriginalLabel().getCanonicalForm()))
        .append('\t')
        .append(checksum == null ? "none" : checksum)
        .append('\t')
        .append(hash == null ? "" : BaseEncoding.base16().lowerCase().encode(hash))
        .append('\t')
        .append(actual == root ? "" : internalToUnicode(actual.getLabel().getCanonicalForm()))
        .append(options.getLineTerminator());
  }

  // Intern by ActionLookupKey while walking. Later operations can use the identity of a single
  // shared object per configured target and access its stable name without another map lookup.
  private static final class HashTarget extends CompactConfiguredHashes.Node<HashTarget> {
    final CqueryNode node;
    int lastParentOrdinal = -1;

    HashTarget(int ordinal, CqueryNode node, boolean deferName) throws IOException {
      this(ordinal, node);
      if (!deferName) {
        name = stableName(node);
      }
    }

    HashTarget(int ordinal, CqueryNode node) {
      super(ordinal, "");
      this.node = node;
    }

    private static String stableName(CqueryNode node) throws IOException {
      String checksum = node.getConfigurationChecksum();
      String identity =
          node.getOriginalLabel().getUnambiguousCanonicalForm()
              + " "
              + (checksum == null ? "null" : checksum);
      if (node.getLookupKey() instanceof ConfiguredTargetKey key) {
        if (key.getExecutionPlatformLabel() == null
            && key.shouldApplyRuleTransition()
            && key.getLabel().equals(node.getOriginalLabel())) {
          return identity;
        }
        Fingerprint fp = new Fingerprint();
        addConfiguredKey(fp, key);
        return identity + (char) 0 + BaseEncoding.base16().lowerCase().encode(fp.digestAndReset());
      }
      if (node.getLookupKey() instanceof AspectKey key) {
        Fingerprint fp = new Fingerprint();
        addAspectKey(fp, key);
        return identity
            + " "
            + key.getAspectClass().getName()
            + (char) 0
            + BaseEncoding.base16().lowerCase().encode(fp.digestAndReset());
      }
      throw new IOException("Unsupported configured hash key: " + node.getLookupKey().getClass());
    }

    private static void addConfiguredKey(Fingerprint fp, ConfiguredTargetKey key) {
      fp.addString(key.getLabel().getUnambiguousCanonicalForm())
          .addNullableString(
              key.getConfigurationKey() == null
                  ? null
                  : key.getConfigurationKey().getOptions().checksum())
          .addNullableString(
              key.getExecutionPlatformLabel() == null
                  ? null
                  : key.getExecutionPlatformLabel().getUnambiguousCanonicalForm())
          .addBoolean(key.shouldApplyRuleTransition());
    }

    private static void addAspectKey(Fingerprint fp, AspectKey key) throws IOException {
      addConfiguredKey(fp, key.getBaseConfiguredTargetKey());
      switch (key.getAspectClass()) {
        case StarlarkAspectClass starlark ->
            fp.addString("starlark")
                .addString(starlark.getExtensionLabel().getUnambiguousCanonicalForm())
                .addString(starlark.getExtensionKey().loadKindForFingerprint())
                .addString(starlark.getExportedName());
        case NativeAspectClass nativeAspect ->
            fp.addString("native").addString(nativeAspect.getClass().getName());
        default ->
            throw new IOException("Unsupported aspect class: " + key.getAspectClass().getClass());
      }
      var attributes = key.getParameters().getAttributes();
      fp.addInt(attributes.keySet().size());
      for (String name : ImmutableList.sortedCopyOf(attributes.keySet())) {
        fp.addString(name).addStrings(attributes.get(name));
      }
      // Base aspects are canonicalized by Bazel on construction, but its presentation-based
      // comparator can tie. Sort their structured digests instead of inheriting that tie order.
      var bases = key.getBaseKeys();
      fp.addInt(bases.size());
      if (!bases.isEmpty()) {
        var identities = new ArrayList<String>(bases.size());
        for (AspectKey base : bases) {
          Fingerprint child = new Fingerprint();
          addAspectKey(child, base);
          identities.add(BaseEncoding.base16().lowerCase().encode(child.digestAndReset()));
        }
        identities.sort(String::compareTo);
        for (String identity : identities) {
          fp.addString(identity);
        }
      }
    }
  }

  private static final Comparator<HashTarget> BY_NAME = Comparator.comparing(target -> target.name);

  private List<HashTarget> collectGraph(
      Iterable<CqueryNode> partialResult,
      List<HashTarget> roots,
      Set<HashTarget> failed,
      @Nullable ExecutorService pool)
      throws IOException, InterruptedException {
    if (pool != null) {
      return collectGraphWithParallelInterning(partialResult, roots, failed, pool);
    }
    Map<ActionLookupKey, HashTarget> seen = new HashMap<>();
    collectRoots(partialResult, roots, seen, /* deferName= */ false);
    List<HashTarget> nodes = new ArrayList<>(roots);
    for (int i = 0; i < nodes.size(); ) {
      int end = nodes.size();
      while (i < end) {
        HashTarget target = nodes.get(i++);
        Iterable<CqueryNode> directDeps =
            depsRetriever.getDirectDeps(target.node, aspectObserver, /* tryEmitInternedDependency= */ null);
        if (directDeps == null) {
          failed.add(target);
          continue;
        }
        ActionLookupKey parentKey = target.node.getLookupKey();
        ImmutableList.Builder<HashTarget> ordered = ImmutableList.builder();
        for (CqueryNode dep : directDeps) {
          ActionLookupKey key = dep.getLookupKey();
          if (key.equals(parentKey)) {
            continue;
          }
          HashTarget child = seen.get(key);
          if (child == null) {
            child = new HashTarget(nodes.size(), dep, /* deferName= */ false);
            seen.put(key, child);
            nodes.add(child);
          }
          if (child.lastParentOrdinal != target.ordinal) {
            child.lastParentOrdinal = target.ordinal;
            ordered.add(child);
          }
        }
        target.deps = ordered.build();
        sortAndCheck(target, /* deduplicate= */ false);
      }
      reportProgress(i, nodes.size());
    }
    return nodes;
  }

  private void collectRoots(
      Iterable<CqueryNode> partialResult,
      List<HashTarget> roots,
      Map<ActionLookupKey, HashTarget> seen,
      boolean deferName)
      throws IOException {
    for (CqueryNode node : partialResult) {
      ActionLookupKey key = node.getLookupKey();
      if ((topLevelConfigurations == null
              || topLevelConfigurations.containsConfiguredTargetKey(key))
          && !seen.containsKey(key)) {
        HashTarget target = new HashTarget(roots.size(), node, deferName);
        seen.put(key, target);
        roots.add(target);
      }
    }
  }

  private List<HashTarget> collectGraphWithParallelInterning(
      Iterable<CqueryNode> partialResult,
      List<HashTarget> roots,
      Set<HashTarget> failed,
      ExecutorService pool)
      throws IOException, InterruptedException {
    var seen = new ConcurrentHashMap<ActionLookupKey, HashTarget>();
    collectRoots(partialResult, roots, seen, /* deferName= */ true);
    var nodes = new ArrayList<>(roots);
    for (int start = 0; start < nodes.size(); ) {
      int end = nodes.size();
      var additions = new ConcurrentLinkedQueue<List<HashTarget>>();
      var failures = new ConcurrentLinkedQueue<HashTarget>();
      ranges(
          pool,
          start,
          end,
          128,
          (a, b) -> {
            var discovered = new ArrayList<HashTarget>();
            class DependencyRow implements Predicate<SkyKey> {
              final ArrayList<HashTarget> deps = new ArrayList<>();
              @Nullable HashTarget parent;

              // A canonical key already in this walk identifies an available configured node.
              @Override
              public boolean test(SkyKey key) {
                HashTarget child = seen.get(key);
                if (child == null) {
                  return false;
                }
                if (child != parent) {
                  deps.add(child);
                }
                return true;
              }
            }
            var row = new DependencyRow();
            for (int i = a; i < b; i++) {
              HashTarget target = nodes.get(i);
              row.parent = target;
              row.deps.clear();
              // Intern newly resolved nodes only after the entire row is available.
              Iterable<CqueryNode> directDeps =
                  depsRetriever.getDirectDeps(target.node, aspectObserver, row);
              if (directDeps == null) {
                failures.add(target);
                continue;
              }
              ActionLookupKey parentKey = target.node.getLookupKey();
              for (CqueryNode dep : directDeps) {
                ActionLookupKey key = dep.getLookupKey();
                if (key.equals(parentKey)) {
                  continue;
                }
                HashTarget child = seen.get(key);
                if (child == null) {
                  child =
                      seen.computeIfAbsent(
                          key,
                          unused -> {
                            var added = new HashTarget(/* ordinal= */ -1, dep);
                            discovered.add(added);
                            return added;
                          });
                }
                row.deps.add(child);
              }
              target.deps = ImmutableList.copyOf(row.deps);
            }
            if (!discovered.isEmpty()) {
              additions.add(discovered);
            }
          });
      // Ordinals index nodes during direct hashing and combination. Only the roots' input
      // order and each row's stable-name order affect the emitted result.
      for (var discovered : additions) {
        for (HashTarget target : discovered) {
          target.ordinal = nodes.size();
          nodes.add(target);
        }
      }
      failed.addAll(failures);
      reportProgress(end, nodes.size());
      start = end;
    }
    finishParallelGraph(nodes, pool);
    return nodes;
  }

  private void reportProgress(int visited, int total) {
    eventHandler.handle(
        Event.progress(
            String.format(
                "Hashing: visited %d configured targets (%d pending)", visited, total - visited)));
  }

  private static void finishParallelGraph(List<HashTarget> nodes, ExecutorService pool)
      throws IOException, InterruptedException {
    ranges(
        pool,
        0,
        nodes.size(),
        1024,
        (a, b) -> {
          for (int i = a; i < b; i++) {
            HashTarget target = nodes.get(i);
            target.name = HashTarget.stableName(target.node);
          }
        });
    ranges(
        pool,
        0,
        nodes.size(),
        512,
        (a, b) -> {
          for (int i = a; i < b; i++) {
            sortAndCheck(nodes.get(i), /* deduplicate= */ true);
          }
        });
  }

  private static void sortAndCheck(HashTarget target, boolean deduplicate) throws IOException {
    if (target.deps.size() <= 1) {
      return;
    }
    var sorted = ImmutableList.sortedCopyOf(BY_NAME, target.deps);
    ImmutableList.Builder<HashTarget> unique = null;
    for (int j = 1; j < sorted.size(); j++) {
      HashTarget previous = sorted.get(j - 1);
      HashTarget current = sorted.get(j);
      if (deduplicate && previous == current) {
        if (unique == null) {
          unique = ImmutableList.builder();
          unique.addAll(sorted.subList(0, j));
        }
        continue;
      }
      if (previous.name.equals(current.name)) {
        throw new IOException(
            "Distinct configured dependencies have the same hash identity under "
                + target.node.getLookupKey());
      }
      if (unique != null) {
        unique.add(current);
      }
    }
    target.deps = unique == null ? sorted : unique.build();
  }

  // Reuse only values verified in this processOutput call. Cap the retained map and node-to-value
  // references independently of closure size. Eviction changes lookup cost only.
  private static final int DIRECT_WINDOW = 1 << 18;
  private static final int DIRECT_CACHE_LIMIT = 1 << 20;
  private static final int DIRECT_EVAL_BATCH = 16384;

  private static final class DirectDigest {
    final Label label;
    @Nullable byte[] bytes;
    @Nullable String error;
    boolean queued;

    DirectDigest(Label label) {
      this.label = label;
    }
  }

  private void computeDistinctDirectHashes(
      List<HashTarget> nodes, Set<HashTarget> failed, int workers, @Nullable ExecutorService pool)
      throws IOException, InterruptedException {
    var cache =
        new ConcurrentHashMap<Label, DirectDigest>(Math.min(nodes.size(), DIRECT_CACHE_LIMIT));
    var context =
        EvaluationContext.newBuilder()
            .setParallelism(workers)
            .setEventHandler(eventHandler)
            .build();
    if (!nodes.isEmpty()) {
      // Flush any pending Skyframe invalidation before consulting done values directly.
      SkyKey barrierKey =
          DirectTargetHashFunction.configuredKey(nodes.get(0).node.getOriginalLabel());
      skyframeExecutor.prepareAndGet(ImmutableSet.of(barrierKey), context);
    }
    for (int first = 0; first < nodes.size(); first += DIRECT_WINDOW) {
      int count = Math.min(DIRECT_WINDOW, nodes.size() - first);
      if (cache.size() + count > DIRECT_CACHE_LIMIT) {
        cache.clear();
      }
      var slots = new DirectDigest[count];
      final int offset = first;
      runDirectRanges(
          pool,
          0,
          count,
          (a, b) -> {
            for (int i = a; i < b; i++) {
              var label = nodes.get(offset + i).node.getOriginalLabel();
              slots[i] = cache.computeIfAbsent(label, DirectDigest::new);
            }
          });
      // Submit fresh labels in their original order.
      var fresh = new ArrayList<DirectDigest>();
      for (DirectDigest slot : slots) {
        if (!slot.queued) {
          slot.queued = true;
          fresh.add(slot);
        }
      }
      // A real evaluator call at the beginning flushes any pending invalidations. Subsequent
      // getExistingValue calls accept only completed entries in the current graph. Missing,
      // dirty and errored entries still go through normal keep-going evaluation.
      runDirectRanges(
          pool,
          0,
          fresh.size(),
          (a, b) -> {
            for (int i = a; i < b; i++) {
              DirectDigest slot = fresh.get(i);
              var key = DirectTargetHashFunction.configuredLookupKey(slot.label);
              if (skyframeExecutor.getEvaluator().getExistingValue(key)
                  instanceof DirectTargetHashFunction.Value value) {
                slot.bytes = value.getDirectHash();
              }
            }
          });
      var missing = new ArrayList<DirectDigest>();
      for (DirectDigest slot : fresh) {
        if (slot.bytes == null) {
          missing.add(slot);
        }
      }
      for (int left = 0; left < missing.size(); left += DIRECT_EVAL_BATCH) {
        List<DirectDigest> group =
            missing.subList(left, Math.min(missing.size(), left + DIRECT_EVAL_BATCH));
        var keysBuilder = ImmutableSet.<SkyKey>builderWithExpectedSize(group.size());
        for (DirectDigest slot : group) {
          keysBuilder.add(DirectTargetHashFunction.configuredKey(slot.label));
        }
        var keys = keysBuilder.build();
        EvaluationResult<SkyValue> result = skyframeExecutor.prepareAndGet(keys, context);
        // The distinct labels in the group match the immutable set's insertion order.
        var orderedKeys = keys.asList();
        for (int i = 0; i < group.size(); i++) {
          DirectDigest slot = group.get(i);
          SkyKey key = orderedKeys.get(i);
          var value = (DirectTargetHashFunction.Value) result.get(key);
          if (value != null) {
            slot.bytes = value.getDirectHash();
          } else {
            ErrorInfo error = result.getError(key);
            slot.error =
                error != null && error.getException() != null
                    ? String.valueOf(error.getException().getMessage())
                    : "unknown error";
          }
        }
      }
      var errors = new ConcurrentLinkedQueue<HashTarget>();
      runDirectRanges(
          pool,
          0,
          count,
          (a, b) -> {
            Fingerprint fp = new Fingerprint();
            for (int i = a; i < b; i++) {
              HashTarget target = nodes.get(offset + i);
              DirectDigest slot = slots[i];
              if (slot.bytes == null) {
                errors.add(target);
                continue;
              }
              target.hash = configuredDirectDigest(target, slot.bytes, fp);
              if (target.hash == null) {
                errors.add(target);
              }
            }
          });
      // Preserve original diagnostic order without synchronizing happy-path work.
      for (HashTarget target :
          errors.stream().sorted(Comparator.comparingInt(t -> t.ordinal)).toList()) {
        failed.add(target);
        var slot = slots[target.ordinal - offset];
        eventHandler.handle(
            Event.error(
                slot.bytes == null
                    ? "failed to compute hash for "
                        + target.node.getDescription(labelPrinter)
                        + ": "
                        + slot.error
                    : "Missing genquery output for " + target.node.getOriginalLabel()));
      }
    }
  }

  private static void runDirectRanges(
      @Nullable ExecutorService pool, int first, int limit, RangeWork work)
      throws InterruptedException, IOException {
    if (first >= limit) {
      return;
    }
    if (pool == null) {
      work.run(first, limit);
    } else {
      ranges(pool, first, limit, 1024, work);
    }
  }

  private void computeDirectHashes(List<HashTarget> batch, Set<HashTarget> failed)
      throws InterruptedException {
    ImmutableSet<SkyKey> keys =
        batch.stream()
            .map(target -> DirectTargetHashFunction.configuredKey(target.node.getOriginalLabel()))
            .collect(toImmutableSet());
    EvaluationResult<SkyValue> result =
        skyframeExecutor.prepareAndGet(
            keys,
            EvaluationContext.newBuilder()
                .setParallelism(SkyQueryEnvironment.DEFAULT_THREAD_COUNT)
                .setEventHandler(eventHandler)
                .build());
    for (HashTarget target : batch) {
      CqueryNode node = target.node;
      var key = DirectTargetHashFunction.configuredKey(node.getOriginalLabel());
      var value = (DirectTargetHashFunction.Value) result.get(key);
      if (value == null) {
        failed.add(target);
        ErrorInfo error = result.getError(key);
        eventHandler.handle(
            Event.error(
                String.format(
                    "failed to compute hash for %s: %s",
                    node.getDescription(labelPrinter),
                    error != null && error.getException() != null
                        ? error.getException().getMessage()
                        : "unknown error")));
        continue;
      }
      target.hash = configuredDirectDigest(target, value.getDirectHash(), new Fingerprint());
      if (target.hash == null) {
        failed.add(target);
        eventHandler.handle(Event.error("Missing genquery output for " + node.getOriginalLabel()));
      }
    }
  }

  @Nullable
  private byte[] configuredDirectDigest(HashTarget target, byte[] direct, Fingerprint fp)
      throws InterruptedException {
    var node = target.node;
    fp.addBytes(direct);
    AnalysisRepoMappingDigest repoMappingDigest = null;
    if (node instanceof RuleConfiguredTarget rule) {
      if (rule.getRuleClassId().key().equals("genquery") && !addGenQueryResult(node, fp)) {
        fp.digestAndReset();
        return null;
      }
      repoMappingDigest = rule.getProvider(AnalysisRepoMappingDigest.class);
    } else if (node instanceof AspectKey key) {
      repoMappingDigest = aspectMappings.get(key);
    }
    if (repoMappingDigest != null) {
      fp.addString("observed-analysis-repo-mapping-v1").addString(repoMappingDigest.digest());
    }
    var config = node.getConfigurationKey();
    if (config == null) {
      fp.addString("<no configuration>");
    } else {
      var options = config.getOptions();
      var checksums = environmentChecksums.get(options);
      if (checksums == null) {
        checksums = environmentChecksums.computeIfAbsent(options, this::checksumsFor);
      }
      fp.addString(
          checksums.test() != null
                  && node instanceof RuleConfiguredTarget rule
                  && TargetUtils.isTestRuleName(rule.getRuleClassString())
              ? checksums.test()
              : checksums.action());
    }
    if (node.getLookupKey() instanceof AspectKey
        || (node.getLookupKey() instanceof ConfiguredTargetKey configuredKey
            && (configuredKey.getExecutionPlatformLabel() != null
                || !configuredKey.shouldApplyRuleTransition()
                || !configuredKey.getLabel().equals(node.getOriginalLabel())))) {
      fp.addString(target.name);
    }
    return fp.digestAndReset();
  }

  private void observeAspect(AspectKey key, AspectValue aspect) {
    var digest = aspect.getProvider(AnalysisRepoMappingDigest.class);
    if (digest != null) {
      aspectMappings.put(key, digest);
    }
  }

  private record EnvironmentChecksums(String action, @Nullable String test) {}

  private EnvironmentChecksums checksumsFor(BuildOptions options) {
    String checksum = options.checksum();
    var names = new TreeSet<String>();
    var coreOptions = options.get(CoreOptions.class);
    if (coreOptions != null) {
      collectInheritedNames(coreOptions.getActionEnvironment(), names);
    }
    String action = withInheritedEnvironment(checksum, names);
    var testOptions = options.get(TestOptions.class);
    if (testOptions == null) {
      return new EnvironmentChecksums(action, null);
    }
    // A test rule can also generate ordinary actions.
    int actionNameCount = names.size();
    collectInheritedNames(testOptions.getTestEnvironment(), names);
    String test = names.size() > actionNameCount ? withInheritedEnvironment(checksum, names) : null;
    return new EnvironmentChecksums(action, test);
  }

  private static void collectInheritedNames(List<EnvVar> options, Set<String> names) {
    // BuildOptions normalizes each list using last occurrence wins.
    for (EnvVar option : options) {
      if (option instanceof EnvVar.Inherit(String name)) {
        names.add(name);
      }
    }
  }

  private String withInheritedEnvironment(String checksum, Set<String> names) {
    if (names.isEmpty()) {
      return checksum;
    }
    Fingerprint fp = new Fingerprint();
    fp.addString("cquery-explicit-client-environment-v1").addString(checksum).addInt(names.size());
    for (String name : names) {
      fp.addString(name).addNullableString(clientEnvironment.get(name));
    }
    return fp.hexDigestAndReset();
  }

  // Native genquery reads the unconfigured graph. A ConfiguredTarget's dependency edges
  // therefore do not encode its result. Read the same result bytes as the actual generating action.
  private boolean addGenQueryResult(CqueryNode node, Fingerprint fingerprint)
      throws InterruptedException {
    if (!(skyframeExecutor.getEvaluator().getExistingValue(node.getLookupKey())
        instanceof RuleConfiguredTargetValue value)) {
      return false;
    }
    var outputs =
        value.getActions().stream()
            .filter(AnalysisTimeFileContents.class::isInstance)
            .map(AnalysisTimeFileContents.class::cast)
            .toList();
    if (outputs.size() != 1) {
      return false;
    }
    fingerprint.addString("native-genquery-result-v1");
    Fingerprint contents = new Fingerprint();
    outputs.get(0).fingerprintFileContents(contents);
    fingerprint.addBytes(contents.digestAndReset());
    return true;
  }
}
