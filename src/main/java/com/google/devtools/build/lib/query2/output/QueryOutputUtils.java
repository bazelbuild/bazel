// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.query2.output;

import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.BlazeQueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.output.OutputFormatter.UnorderedFormatter;

import java.io.IOException;
import java.io.PrintStream;

/** Static utility methods for outputting a query. */
public class QueryOutputUtils {
  // Utility class cannot be instantiated.
  private QueryOutputUtils() {}

  public static boolean orderResults(QueryOptions queryOptions, OutputFormatter formatter) {
    return queryOptions.orderResults || !(formatter instanceof UnorderedFormatter);
  }

  public static void output(QueryOptions queryOptions, QueryEvalResult<Target> result,
      OutputFormatter formatter, PrintStream outputStream, AspectResolver aspectResolver)
      throws IOException, InterruptedException {
    if (orderResults(queryOptions, formatter)) {
      formatter.output(queryOptions, ((BlazeQueryEvalResult<Target>) result).getResultGraph(),
          outputStream, aspectResolver);
    } else {
      ((UnorderedFormatter) formatter).outputUnordered(queryOptions, result.getResultSet(),
          outputStream, aspectResolver);
    }
  }
}
