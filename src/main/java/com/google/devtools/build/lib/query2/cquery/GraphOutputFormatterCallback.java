// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.query.output.GraphOutputWriter;
import com.google.devtools.build.lib.query2.query.output.GraphOutputWriter.NodeReader;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import java.io.OutputStream;
import java.util.Comparator;

/** cquery output formatter that prints the result as factored graph in AT&amp;T GraphViz format. */
class GraphOutputFormatterCallback extends CqueryThreadsafeCallback {
  @Override
  public String getName() {
    return "graph";
  }

  /** Interface for finding a configured target's direct dependencies. */
  @FunctionalInterface
  public interface DepsRetriever {
    Iterable<KeyedConfiguredTarget> getDirectDeps(KeyedConfiguredTarget target)
        throws InterruptedException;
  }

  private final DepsRetriever depsRetriever;

  private final GraphOutputWriter.NodeReader<KeyedConfiguredTarget> nodeReader =
      new NodeReader<KeyedConfiguredTarget>() {

        private final Comparator<KeyedConfiguredTarget> configuredTargetOrdering =
            (ct1, ct2) -> {
              // Order graph output first by target label, then by configuration hash.
              Label label1 = ct1.getLabel();
              Label label2 = ct2.getLabel();
              if (!label1.equals(label2)) {
                return label1.compareTo(label2);
              }
              String checksum1 = ct1.getConfigurationChecksum();
              String checksum2 = ct2.getConfigurationChecksum();
              if (checksum1 == null) {
                return -1;
              } else if (checksum2 == null) {
                return 1;
              } else {
                return checksum1.compareTo(checksum2);
              }
            };

        @Override
        public String getLabel(Node<KeyedConfiguredTarget> node) {
          // Node payloads are ConfiguredTargets. Output node labels are target labels + config
          // hashes.
          KeyedConfiguredTarget kct = node.getLabel();
          return String.format(
              "%s (%s)", kct.getLabel(), shortId(getConfiguration(kct.getConfigurationKey())));
        }

        @Override
        public Comparator<KeyedConfiguredTarget> comparator() {
          return configuredTargetOrdering;
        }
      };

  GraphOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<KeyedConfiguredTarget> accessor,
      DepsRetriever depsRetriever) {
    super(eventHandler, options, out, skyframeExecutor, accessor);
    this.depsRetriever = depsRetriever;
  }

  @Override
  public void processOutput(Iterable<KeyedConfiguredTarget> partialResult)
      throws InterruptedException {
    // Transform the cquery-backed graph into a Digraph to make it suitable for GraphOutputWriter.
    // Note that this involves an extra iteration over the entire query result subgraph. We could
    // conceptually merge transformation and output writing into the same iteration if needed.
    Digraph<KeyedConfiguredTarget> graph = new Digraph<>();
    ImmutableSet<KeyedConfiguredTarget> allNodes = ImmutableSet.copyOf(partialResult);
    for (KeyedConfiguredTarget configuredTarget : partialResult) {
      Node<KeyedConfiguredTarget> node = graph.createNode(configuredTarget);
      for (KeyedConfiguredTarget dep : depsRetriever.getDirectDeps(configuredTarget)) {
        if (allNodes.contains(dep)) {
          Node<KeyedConfiguredTarget> depNode = graph.createNode(dep);
          graph.addEdge(node, depNode);
        }
      }
    }

    GraphOutputWriter<KeyedConfiguredTarget> graphWriter =
        new GraphOutputWriter<>(
            nodeReader,
            options.getLineTerminator(),
            /*sortLabels=*/ true,
            options.graphNodeStringLimit,
            // select() conditions don't matter for cquery because cquery operates post-analysis
            // phase, when select()s have been resolved and removed from the graph.
            /*maxConditionalEdges=*/ 0,
            options.graphFactored);
    graphWriter.write(graph, /*conditionalEdges=*/ null, outputStream);
  }
}
