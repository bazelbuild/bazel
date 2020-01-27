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
package com.google.devtools.build.skyframe;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.devtools.build.lib.events.Event;

/** Filters out events which should not be stored during evaluation in {@link ParallelEvaluator}. */
public interface EventFilter extends Predicate<Event> {
  /**
   * Returns true if any events/postables should be stored. Otherwise, optimizations may be made to
   * avoid doing unnecessary work when evaluating node entries.
   */
  boolean storeEventsAndPosts();

  default Predicate<SkyKey> depEdgeFilterForEventsAndPosts(SkyKey primaryKey) {
    return Predicates.alwaysTrue();
  }
}
