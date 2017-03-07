// Copyright 2017 The Bazel Authors. All rights reserved.
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

/** Marker interface for a {@link ThreadSafe} {@link MinDepthUniquifier}. */
@ThreadSafe
public interface ThreadSafeMinDepthUniquifier<T> extends MinDepthUniquifier<T> {
  // There's a natural benign check-then-act race in all concurrent uses of this interface. Thread
  // T1 may think it's about to be the first one to process an element at a depth no greater than
  // d1. But before t1 finishes processing the element, Thread T2 may think _it's_ about to be first
  // one to process an element at a depth no greater than d2. If d2 < d1, then T1's work is probably
  // wasted.
}
