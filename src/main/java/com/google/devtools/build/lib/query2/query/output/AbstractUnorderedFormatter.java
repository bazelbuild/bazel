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
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.common.CommonQueryOptions;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver;
import com.google.devtools.build.lib.query2.query.output.QueryOptions.OrderOutput;
import java.io.IOException;
import java.io.OutputStream;

abstract class AbstractUnorderedFormatter extends OutputFormatter implements StreamedFormatter {

  @Override
  public void setOptions(CommonQueryOptions options, AspectResolver aspectResolver) {}

  @Override
  public void output(
      QueryOptions options, Digraph<Target> result, OutputStream out, AspectResolver aspectResolver)
      throws IOException, InterruptedException {
    setOptions(options, aspectResolver);
    OutputFormatterCallback.processAllTargets(
        createPostFactoStreamCallback(out, options), getOrderedTargets(result, options));
  }

  protected Iterable<Target> getOrderedTargets(Digraph<Target> result, QueryOptions options) {
    Iterable<Node<Target>> orderedResult =
        options.orderOutput == OrderOutput.DEPS
            ? result.getTopologicalOrder()
            : result.getTopologicalOrder(new FormatUtils.TargetOrdering());
    return Iterables.transform(orderedResult, Node::getLabel);
  }
}