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

import com.google.common.collect.Iterables;
import com.google.common.hash.HashFunction;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.common.CommonQueryOptions;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver;
import com.google.devtools.build.lib.query2.query.output.QueryOptions.OrderOutput;
import java.io.IOException;
import java.io.OutputStream;
import javax.annotation.Nullable;

abstract class AbstractUnorderedFormatter extends OutputFormatter implements StreamedFormatter {

  @Override
  public void setOptions(
      CommonQueryOptions options, AspectResolver aspectResolver, HashFunction hashFunction) {}

  /** Optionally sets a handler for reporting status output / errors. */
  @Override
  public void setEventHandler(@Nullable EventHandler eventHandler) {}

  @Override
  public void output(
      QueryOptions options,
      Digraph<Target> result,
      OutputStream out,
      AspectResolver aspectResolver,
      @Nullable EventHandler eventHandler,
      HashFunction hashFunction,
      LabelPrinter labelPrinter)
      throws IOException, InterruptedException {
    setOptions(options, aspectResolver, hashFunction);
    setEventHandler(eventHandler);
    OutputFormatterCallback.processAllTargets(
        createPostFactoStreamCallback(out, options, labelPrinter),
        getOrderedTargets(result, options));
  }

  protected Iterable<Target> getOrderedTargets(Digraph<Target> result, QueryOptions options) {
    if (options.orderOutput == OrderOutput.FULL) {
      // Get targets in total order, the difference here from topological ordering is the sorting of
      // nodes before post-order visitation (which ensures determinism at a time cost).
      return Iterables.transform(
          result.getTopologicalOrder(new FormatUtils.TargetOrdering()), Node::getLabel);
    } else if (options.orderOutput == OrderOutput.DEPS) {
      // Get targets in topological order.
      return Iterables.transform(result.getTopologicalOrder(), Node::getLabel);
    }
    return result.getLabels();
  }
}
