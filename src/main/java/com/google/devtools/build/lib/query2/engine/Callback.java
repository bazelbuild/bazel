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
package com.google.devtools.build.lib.query2.engine;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.BatchCallback;
import com.google.devtools.build.lib.util.ThreadSafeBatchCallback;

/**
 * Query callback to be called by a {@link QueryExpression} when it has part of the computation
 * result. Assuming the {@code QueryEnvironment} supports it, it would allow the caller
 * to stream the results.
 */
@ThreadSafe
public interface Callback<T> extends ThreadSafeBatchCallback<T, QueryException> {

  /**
   * According to the {@link BatchCallback} interface, repeated elements may be passed in here.
   * However, {@code QueryExpression}s calling the callback do not need to maintain this property,
   * as the {@code QueryEnvironment} should filter out duplicates.
   */
  @Override
  void process(Iterable<T> partialResult) throws QueryException, InterruptedException;
}
