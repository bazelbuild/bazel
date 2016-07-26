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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

import java.util.Map;

/**
 * A graph that exposes thin representations of its entries and structure, for use during
 * invalidation.
 *
 * <p>Public only for use in alternative graph implementations.
 */
@ThreadSafe
public interface InvalidatableGraph {
  /**
   * Fetches all the given thin nodes. Returns a map {@code m} such that, for all {@code k} in
   * {@code keys}, {@code m.get(k).equals(e)} iff {@code get(k) == e} and {@code e != null}, and
   * {@code !m.containsKey(k)} iff {@code get(k) == null}.
   */
  Map<SkyKey, ? extends ThinNodeEntry> getBatchForInvalidation(Iterable<SkyKey> keys);
}
