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

import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.common.CommonQueryOptions;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver;
import java.io.OutputStream;
import javax.annotation.Nullable;

/**
 * Unordered streamed output formatter (wrt. dependency ordering).
 *
 * <p>Formatters that support streamed output may be used when only the set of query results is
 * requested but their ordering is irrelevant.
 *
 * <p>The benefit of using a streamed formatter is that we can save the potentially expensive
 * subgraph extraction step before presenting the query results and that depending on the query
 * environment used, it can be more memory performant, as it does not aggregate all the data
 * before writing in the output.
 */
public interface StreamedFormatter {
  /** Specifies options to be used by subsequent calls to {@link #createStreamCallback}. */
  void setOptions(CommonQueryOptions options, AspectResolver aspectResolver);

  /** Sets an optional handler for reporting status output / errors. */
  void setEventHandler(@Nullable EventHandler eventHandler);

  /**
   * Returns a {@link ThreadSafeOutputFormatterCallback} whose
   * {@link OutputFormatterCallback#process} outputs formatted {@link Target}s to the given
   * {@code out}.
   *
   * <p>Takes any options specified via the most recent call to {@link #setOptions} into
   * consideration.
   *
   * <p>Intended to be use for streaming out during evaluation of a query.
   */
  ThreadSafeOutputFormatterCallback<Target> createStreamCallback(
      OutputStream out, QueryOptions options, QueryEnvironment<?> env);

  /**
   * Same as {@link #createStreamCallback}, but intended to be used for outputting the
   * already-computed result of a query.
   */
  OutputFormatterCallback<Target> createPostFactoStreamCallback(
      OutputStream out, QueryOptions options);
}