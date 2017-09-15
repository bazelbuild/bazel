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

/**
 * An interface for the evaluator for a particular graph version.
 */
public interface Evaluator {
  /**
   * Factory to create Evaluator instances.
   */
  interface Factory {
    /**
     * @param graph the graph to operate on
     * @param graphVersion the version at which to write entries in the graph.
     * @param reporter where to write warning/error/progress messages.
     * @param keepGoing whether {@link #eval} should continue if building a {link Value} fails.
     *                  Otherwise, we throw an exception on failure.
     */
    Evaluator create(
        ProcessableGraph graph, long graphVersion, EventHandler reporter, boolean keepGoing);
  }

  /**
   * Evaluates a set of values. Returns an {@link EvaluationResult}. All elements of skyKeys must
   * be keys for Values of subtype T.
   */
  <T extends SkyValue> EvaluationResult<T> eval(Iterable<? extends SkyKey> skyKeys)
      throws InterruptedException;
}
