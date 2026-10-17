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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.query2.SkyQueryEnvironment;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.cquery.GraphOutputFormatterCallback.DepsRetriever;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.query.hash.DirectTargetHashFunction;
import com.google.devtools.build.lib.query2.query.hash.TransitiveHashes;
import com.google.devtools.build.lib.server.FailureDetails.Query;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * cquery output formatter for {@code --generate_hashes}. Prints each configured target in the
 * result followed by a deterministic hash of it and its transitive configured dependencies.
 *
 * <p>Unlike {@code query --generate_hashes}, the graph that is hashed is the configured target
 * graph: {@code select()}s are resolved, toolchains and execution platforms are the ones that were
 * actually chosen, and each node is hashed together with its configuration. A target's own hash is
 * the {@link DirectTargetHashFunction} hash of its unconfigured target (attribute values, rule
 * definition, or source file contents), combined with the checksum of its build options.
 */
class HashesOutputFormatterCallback extends CqueryThreadsafeCallback {

  static final String NAME = "generate_hashes";

  private final SkyframeExecutor skyframeExecutor;
  private final DepsRetriever depsRetriever;
  private final LabelPrinter labelPrinter;

  HashesOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<CqueryNode> accessor,
      DepsRetriever depsRetriever,
      LabelPrinter labelPrinter) {
    super(eventHandler, options, out, skyframeExecutor, accessor, /* uniquifyResults= */ false);
    this.skyframeExecutor = skyframeExecutor;
    this.depsRetriever = depsRetriever;
    this.labelPrinter = labelPrinter;
  }

  @Override
  public String getName() {
    return NAME;
  }

  private int failures = 0;

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
    Map<ActionLookupKey, CqueryNode> roots = new LinkedHashMap<>();
    for (CqueryNode node : partialResult) {
      roots.putIfAbsent(node.getLookupKey(), node);
    }

    // Walk the configured graph breadth-first, recording each node's deps.
    Map<ActionLookupKey, CqueryNode> nodes = new HashMap<>(roots);
    Map<ActionLookupKey, List<ActionLookupKey>> deps = new HashMap<>();
    List<CqueryNode> frontier = new ArrayList<>(roots.values());
    while (!frontier.isEmpty()) {
      List<CqueryNode> next = new ArrayList<>();
      for (CqueryNode node : frontier) {
        List<ActionLookupKey> nodeDeps = new ArrayList<>();
        for (CqueryNode dep : depsRetriever.getDirectDeps(node)) {
          ActionLookupKey depKey = dep.getLookupKey();
          if (depKey.equals(node.getLookupKey())) {
            continue;
          }
          nodeDeps.add(depKey);
          if (nodes.putIfAbsent(depKey, dep) == null) {
            next.add(dep);
          }
        }
        deps.put(node.getLookupKey(), nodeDeps);
      }
      frontier = next;
      eventHandler.handle(
          Event.progress(
              String.format(
                  "Hashing: visited %d configured targets (%d pending)",
                  deps.size(), frontier.size())));
    }

    // The direct hashes of the unconfigured targets are computed, and memoized, by Skyframe.
    ImmutableSet<SkyKey> keys =
        nodes.values().stream()
            .map(node -> DirectTargetHashFunction.key(node.getOriginalLabel()))
            .collect(toImmutableSet());
    EvaluationResult<SkyValue> result =
        skyframeExecutor.prepareAndGet(
            keys,
            EvaluationContext.newBuilder()
                .setParallelism(SkyQueryEnvironment.DEFAULT_THREAD_COUNT)
                .setEventHandler(eventHandler)
                .build());

    Map<ActionLookupKey, String> names = new HashMap<>();
    nodes.forEach((key, node) -> names.put(key, name(node)));
    Comparator<ActionLookupKey> byName = Comparator.comparing(names::get);
    Map<ActionLookupKey, TransitiveHashes.Node<ActionLookupKey>> graph = new HashMap<>();
    for (Map.Entry<ActionLookupKey, CqueryNode> entry : nodes.entrySet()) {
      CqueryNode node = entry.getValue();
      var value =
          (DirectTargetHashFunction.Value)
              result.get(DirectTargetHashFunction.key(node.getOriginalLabel()));
      if (value == null) {
        continue; // Hashed as a missing leaf by its dependents, reported below for roots.
      }
      Fingerprint fp = new Fingerprint().addBytes(value.getDirectHash());
      String checksum = node.getConfigurationChecksum();
      fp.addString(checksum == null ? "<no configuration>" : checksum);
      if (node.getLookupKey() instanceof AspectKey aspectKey) {
        fp.addString(aspectKey.getDescription());
      }
      graph.put(
          entry.getKey(),
          new TransitiveHashes.Node<>(
              fp.digestAndReset(),
              deps.get(entry.getKey()).stream()
                  .distinct()
                  .sorted(byName)
                  .collect(ImmutableList.toImmutableList())));
    }

    eventHandler.handle(
        Event.progress(String.format("Hashing: combining hashes of %d targets", graph.size())));
    Map<ActionLookupKey, byte[]> hashes = TransitiveHashes.combine(graph, byName, names::get);
    for (CqueryNode root : roots.values()) {
      String description =
          String.format(
              "%s (%s)",
              root.getDescription(labelPrinter),
              shortId(getConfiguration(root.getConfigurationKey())));
      byte[] hash = hashes.get(root.getLookupKey());
      if (hash != null) {
        // Written directly instead of through addResult, because a failure below skips close().
        printStream
            .append(description)
            .append(' ')
            .append(BaseEncoding.base16().lowerCase().encode(hash))
            .append(options.getLineTerminator());
        continue;
      }
      failures++;
      ErrorInfo error = result.getError(DirectTargetHashFunction.key(root.getOriginalLabel()));
      eventHandler.handle(
          Event.error(
              String.format(
                  "failed to compute hash for %s: %s",
                  description,
                  error != null && error.getException() != null
                      ? error.getException().getMessage()
                      : "unknown error")));
    }
    printStream.flush();
  }

  /** A stable name of a configured node, identifying it among all nodes of the graph. */
  private static String name(CqueryNode node) {
    Label label = node.getOriginalLabel();
    String checksum = node.getConfigurationChecksum();
    String name =
        label.getUnambiguousCanonicalForm() + " " + (checksum == null ? "null" : checksum);
    if (node.getLookupKey() instanceof AspectKey aspectKey) {
      name += " " + aspectKey.getDescription();
    }
    return name;
  }
}
