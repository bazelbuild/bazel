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

/**
 * Query callback to be called by a {@link QueryExpression} when it has part of the computation
 * result. Assuming the {@code QueryEnvironment} supports it, it would allow the caller
 * to stream the results.
 */
public interface Callback<T> {

  /**
   * Called by {@code QueryExpression} when it has been able to compute part of the result.
   *
   * <p>Note that this method can be called several times for a QueryExpression. Callers
   * implementing this method should assume that multiple calls can happen.
   *
   * @param partialResult Part of the result. Note that from the caller's perspective, it is
   * guaranteed that no repeated elements will be returned. However {@code QueryExpression}s calling
   * the callback do not need to maintain this property, as the {@code QueryEnvironment} should
   * handle the uniqueness.
   */
  void process(Iterable<T> partialResult) throws QueryException, InterruptedException;
}
