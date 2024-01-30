// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.query.output;

import com.google.common.hash.HashFunction;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver;
import com.google.devtools.build.lib.query2.query.output.FormatUtils.TargetOrdering;
import com.google.devtools.build.lib.query2.query.output.GraphOutputWriter.NodeReader;
import com.google.devtools.build.lib.query2.query.output.QueryOptions.OrderOutput;
import java.io.OutputStream;
import java.util.Comparator;

/**
 * An output formatter that prints the result as factored graph in AT&amp;T
 * GraphViz format.
 */
class GraphOutputFormatter extends OutputFormatter {
  @Override
  public String getName() {
    return "graph";
  }

  private static final GraphOutputWriter.NodeReader<Target> NODE_READER =
      new NodeReader<Target>() {
        private final TargetOrdering targetOrdering = new FormatUtils.TargetOrdering();

        @Override
        public String getLabel(Node<Target> node, LabelPrinter labelPrinter) {
          // Node payloads are Targets. Output node labels are target labels.
          return labelPrinter.toString(node.getLabel().getLabel());
        }

        @Override
        public Comparator<Target> comparator() {
          return targetOrdering;
        }
      };

  @Override
  public void output(
      QueryOptions options,
      Digraph<Target> result,
      OutputStream out,
      AspectResolver aspectProvider,
      EventHandler eventHandler,
      HashFunction hashFunction,
      LabelPrinter labelPrinter) {
    boolean sortLabels = options.orderOutput == OrderOutput.FULL;
    GraphOutputWriter<Target> graphWriter =
        new GraphOutputWriter<>(
            NODE_READER,
            options.getLineTerminator(),
            sortLabels,
            options.graphNodeStringLimit,
            options.graphConditionalEdgesLimit,
            options.graphFactored,
            labelPrinter);
    graphWriter.write(result, new ConditionalEdges(result), out);
  }
}
