// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.engine.DigraphQueryEvalResult;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver;
import com.google.devtools.build.lib.query2.query.output.QueryOptions.OrderOutput;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Set;

/** Static utility methods for outputting a query. */
public class QueryOutputUtils {
  // Utility class cannot be instantiated.
  private QueryOutputUtils() {}

  public static boolean shouldStreamResults(QueryOptions queryOptions, OutputFormatter formatter) {
    return (queryOptions.orderOutput == OrderOutput.NO
            || (queryOptions.orderOutput == OrderOutput.AUTO && queryOptions.preferUnorderedOutput))
        && formatter instanceof StreamedFormatter;
  }

  public static void output(QueryOptions queryOptions, QueryEvalResult result,
      Set<Target> targetsResult, OutputFormatter formatter, OutputStream outputStream,
      AspectResolver aspectResolver)
      throws IOException, InterruptedException {
    /*
     * This is not really streaming, but we are using the streaming interface for writing into the
     * output everything in one batch. This happens when the QueryEnvironment does not
     * support streaming but we don't care about ordered results.
     */
    if (shouldStreamResults(queryOptions, formatter)) {
      StreamedFormatter streamedFormatter = (StreamedFormatter) formatter;
      streamedFormatter.setOptions(queryOptions, aspectResolver);
      OutputFormatterCallback.processAllTargets(
          streamedFormatter.createPostFactoStreamCallback(outputStream, queryOptions),
          targetsResult);
    } else {
      @SuppressWarnings("unchecked")
      DigraphQueryEvalResult<Target> digraphQueryEvalResult =
          (DigraphQueryEvalResult<Target>) result;
      Digraph<Target> subgraph;

      try (SilentCloseable closeable = Profiler.instance().profile("digraph.extractSubgraph")) {
        subgraph = digraphQueryEvalResult.getGraph().extractSubgraph(targetsResult);
      }

      try (SilentCloseable closeable = Profiler.instance().profile("formatter.output")) {
        formatter.output(queryOptions, subgraph, outputStream, aspectResolver);
      }
    }
  }
}
