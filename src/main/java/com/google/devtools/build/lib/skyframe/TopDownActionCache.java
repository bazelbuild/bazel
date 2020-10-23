// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.actionsketch.ActionSketch;
import javax.annotation.Nullable;

/**
 * A top-down action cache is a cache of {@link ActionSketch} to {@link ActionExecutionValue}.
 *
 * <p>Unlike {@link com.google.devtools.build.lib.actions.ActionCacheChecker}, a top-down cache can
 * cull large subgraphs by computing the transitive cache key (known as the {@link ActionSketch}).
 */
public interface TopDownActionCache {

  /** Retrieves the cached value for the given action sketch, or null. */
  @Nullable
  ActionExecutionValue get(ActionSketch sketch);

  /** Puts the sketch into the top-down cache. May complete asynchronously. */
  void put(ActionSketch sketch, ActionExecutionValue value);
}
