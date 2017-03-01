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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.concurrent.ForkJoinPool;

/** Listener for calls to the internal methods of {@link QueryExpression} used for evaluation. */
@ThreadSafe
public interface QueryExpressionEvalListener<T> {
  /** Called right before {@link QueryExpression#evalImpl} is called. */
  void onEval(
      QueryExpression expr,
      QueryEnvironment<T> env,
      VariableContext<T> context,
      Callback<T> callback);

  /** Called right before {@link QueryExpression#parEvalImpl} is called. */
  void onParEval(
      QueryExpression expr,
      QueryEnvironment<T> env,
      VariableContext<T> context,
      ThreadSafeCallback<T> callback,
      ForkJoinPool forkJoinPool);

  /** A {@link QueryExpressionEvalListener} that does nothing. */
  class NullListener<T> implements QueryExpressionEvalListener<T> {
    private static final NullListener<?> INSTANCE = new NullListener<>();

    private NullListener() {
    }

    @SuppressWarnings("unchecked")
    public static <T> NullListener<T> instance() {
      return (NullListener<T>) INSTANCE;
    }

    @Override
    public void onEval(
        QueryExpression expr,
        QueryEnvironment<T> env,
        VariableContext<T> context,
        Callback<T> callback) {
    }

    @Override
    public void onParEval(
        QueryExpression expr,
        QueryEnvironment<T> env,
        VariableContext<T> context,
        ThreadSafeCallback<T> callback,
        ForkJoinPool forkJoinPool) {
    }
  }
}

