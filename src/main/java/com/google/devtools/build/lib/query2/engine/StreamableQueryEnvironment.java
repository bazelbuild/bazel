// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.engine;

import com.google.common.base.Predicate;
import java.util.List;

/**
 * The environment of a Blaze query which supports predefined streaming operations.
 *
 * @param <T> the node type of the dependency graph
 */
public interface StreamableQueryEnvironment<T> extends QueryEnvironment<T> {

  /** Retrieves and processes all reverse dependencies of given expression in a streaming manner. */
  QueryTaskFuture<Void> getAllRdeps(
      QueryExpression expression,
      Predicate<T> universe,
      VariableContext<T> context,
      Callback<T> callback,
      int depth);

  /** Similar to {@link #getAllRdeps} but finds all rdeps without a depth bound. */
  QueryTaskFuture<Void> getAllRdepsUnboundedParallel(
      QueryExpression expression, VariableContext<T> context, Callback<T> callback);

  /**
   * Similar to {@link #getAllRdepsUnboundedParallel} but finds rdeps in a universe without a depth
   * depth.
   *
   * @param expression a "rdeps" expression without depth, such as rdeps(u, x)
   * @param args two-item list containing both universe 'u' and argument set 'x' in rdeps(u, x)
   */
  QueryTaskFuture<Void> getRdepsUnboundedInUniverseParallel(
      QueryExpression expression,
      VariableContext<T> context,
      List<Argument> args,
      Callback<T> callback);
}
