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

package com.google.devtools.build.skyframe;

import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.common.options.OptionsClassProvider;

import javax.annotation.Nullable;

/**
 * A BuildDriver wraps a MemoizingEvaluator, passing along the proper Version.
 */
public interface BuildDriver {
  /**
   * See {@link MemoizingEvaluator#evaluate}, which has the same semantics except for the
   * inclusion of a {@link Version} value.
   */
  <T extends SkyValue> EvaluationResult<T> evaluate(
      Iterable<SkyKey> roots, boolean keepGoing, int numThreads, EventHandler reporter)
      throws InterruptedException;

  /**
   * Retrieve metadata about the computation over the given roots. Data returned is specific to the
   * underlying evaluator implementation.
   */
  String meta(Iterable<SkyKey> roots, OptionsClassProvider options)
      throws AbruptExitException, InterruptedException;


  MemoizingEvaluator getGraphForTesting();

  @Nullable
  SkyValue getExistingValueForTesting(SkyKey key);

  @Nullable
  ErrorInfo getExistingErrorForTesting(SkyKey key);

}
